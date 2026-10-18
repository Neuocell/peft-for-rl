#!/usr/bin/env python3
"""Registered 200->250 continuation with native GPU first-rollout verification."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/root/peft-for-rl')
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.local import continue_gradtop_r32 as cont
from scripts.local import run_gradtop_clean_after_real as clean
from scripts.local import run_real_after_eval as queue
from scripts.local.real_experiment_eval import _load_file


def read(path):
    return json.loads(Path(path).read_text())


def guard(plan):
    clean.assert_plan_unchanged(OPS / 'plan.json', plan['_sha256'])
    queue.verify_code(plan)
    for group in ('source_artifact_sha256', 'resume_source_sha256'):
        for path, expected in plan[group].items():
            clean.assert_plan_unchanged(path, expected)


def checkpoint(plan, step):
    report = clean.validate_complete_checkpoint(plan, step)
    import torch
    torch.set_num_threads(1)
    for rank in range(4):
        path = Path(plan['checkpoint_root']) / f'global_step_{step}/actor/optim_world_size_4_rank_{rank}.pt'
        optimizer = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        states = optimizer['state']
        if len(states) != 392 or {int(s['step']) for s in states.values()} != {4 * step}:
            raise RuntimeError(f'Incorrect Adam state/count at step{step}, rank{rank}')
        if any(float(g['lr']) != 1e-5 for g in optimizer['param_groups']):
            raise RuntimeError('Restored optimizer learning rate differs')
        del optimizer
    return {**report, 'all_four_optimizer_ranks_verified': True}


def budget(plan):
    source = Path(plan['checkpoint_root']) / f"global_step_{plan['source_step']}"
    snapshot = Path(plan['snapshot_root']) / f"step_{plan['source_step']:03d}"
    ckpt_bytes, snap_bytes = cont.directory_bytes(source), cont.directory_bytes(snapshot)
    count = (plan['target_step'] - plan['source_step']) // 10
    required = math.ceil(ckpt_bytes * 1.10 + count * snap_bytes * 1.10) + 2 * 1024**3
    return {'checkpoint_bytes': ckpt_bytes, 'snapshot_bytes': snap_bytes,
            'additional_snapshots': count, 'required_free_bytes': required,
            'available_free_bytes': shutil.disk_usage(ROOT).free,
            'checkpoint_growth_margin': 1.10, 'snapshot_growth_margin': 1.10,
            'headroom_gib': 2, 'retention_credit_bytes': 0,
            'retention': 'Latest two full actors:200/250; older evaluated adapters preserved separately'}


def preserve_adapters(plan):
    entries = {}
    for step in plan['preserve_adapter_steps']:
        source = Path(plan['checkpoint_root']) / f'global_step_{step}/actor/peft_adapter'
        target = OPS / 'preserved_adapters' / f'step_{step}'
        target.mkdir(parents=True, exist_ok=False)
        files = {}
        for path in source.iterdir():
            if not path.is_file():
                raise RuntimeError('Unexpected nested adapter artifact')
            dest = target / path.name
            os.link(path.resolve(), dest)
            files[path.name] = queue.sha256(path)
            clean.assert_plan_unchanged(dest, files[path.name])
        if not {'adapter_config.json', 'adapter_model.safetensors'}.issubset(files):
            raise RuntimeError('Incomplete preserved adapter')
        entries[str(step)] = {'source': str(source), 'backup': str(target), 'files': files}
    queue.write_json(OPS / 'adapter_preservation.json', entries)


def restore_adapters():
    for item in read(OPS / 'adapter_preservation.json').values():
        source, backup = Path(item['source']), Path(item['backup'])
        for name, digest in item['files'].items():
            clean.assert_plan_unchanged(backup / name, digest)
        if not source.exists():
            if source.is_symlink():
                raise RuntimeError('Unexpected dangling original adapter symlink')
            source.parent.mkdir(parents=True, exist_ok=True)
            source.symlink_to(backup, target_is_directory=True)
        for name, digest in item['files'].items():
            clean.assert_plan_unchanged(source / name, digest)


def sync_receipts(plan, required=False):
    module = _load_file('gradtop250_resume_verifier', ROOT / 'verl/utils/lora_resume_verification.py')
    report = module.validate_receipt_directory(
        OPS / 'sync_receipts', expected_world_size=4,
        expected_checkpoint=Path(plan['checkpoint_root']) / f"global_step_{plan['source_step']}/actor",
        required=required)
    if report:
        env = plan['sync_environment']
        if (report['token'] != env['VERL_LORA_RESUME_VERIFY_TOKEN']
                or report['source_file_sha256'] != env['VERL_LORA_RESUME_VERIFY_SHA256']
                or report['tensor_count'] != 392
                or Path(report['source']).resolve() != Path(env['VERL_LORA_RESUME_VERIFY_SOURCE']).resolve()):
            raise RuntimeError('Synchronization receipts do not match this launch/source200')
    return report


def preflight(plan):
    if (plan['source_step'], plan['target_step'], plan['evaluate_terminal']) != (200, 250, True):
        raise RuntimeError('Only the registered200->250+evaluation is supported')
    guard(plan)
    prior = read(plan['predecessor_status'])
    if (prior['status'], prior['experiment'], prior['latest_logged_step']) != ('complete', plan['experiment'], 200):
        raise RuntimeError('Previous training/evaluation not complete')
    reporter = read(plan['predecessor_reporter_status'])
    if reporter['status'] != 'complete':
        raise RuntimeError('Previous report not complete')
    for identity in plan['predecessor_processes']:
        if clean.process_alive(identity):
            raise RuntimeError('Previous owner still running')
    for path, expected in plan['source_checkpoint_sha256'].items():
        clean.assert_plan_unchanged(path, expected)
    cont.ensure_latest(plan, plan['source_step'])
    for path in [Path(plan['checkpoint_root']) / 'global_step_250', clean.evaluation_output(plan, 250)]:
        if path.exists():
            raise RuntimeError(f'Refusing an existing destination:{path}')
    if any(int(p.name[5:]) > 200 for p in Path(plan['snapshot_root']).glob('step_*') if p.name[5:].isdigit()):
        raise RuntimeError('Unexpected post200 snapshot already exists')
    report = checkpoint(plan, 200)
    if report['adapter_sha256'] != plan['sync_environment']['VERL_LORA_RESUME_VERIFY_SHA256']:
        raise RuntimeError('Source adapter mismatch')
    old = read(plan['source_contract_copy'])
    clean.assert_plan_unchanged(plan['source_contract']['path'], plan['source_contract']['sha256'])
    target = cont.protocol_report(plan, 250)
    cont.assert_contract_extension(old, target, 200, 250)
    tool = _load_file('gradtop250_runtime', ROOT / 'scripts/analysis/record_rlvr_runtime.py')
    identity = tool.make_identity(target, read(plan['protocol_file']), ROOT)
    if read(plan['runtime_manifest'])['identity'] != identity:
        raise RuntimeError('Existing runtime identity differs from continuation')
    clean.verify_eval_assets(plan)
    output = clean.evaluation_output(plan, 200)
    clean.verified_evaluation(plan, output, expected_step=200, expected_name=output.name)
    measured = budget(plan)
    if measured['available_free_bytes'] < measured['required_free_bytes']:
        raise RuntimeError('Insufficient measured disk budget')
    queue.write_json(OPS / 'source_checkpoint200_verification.json', report)
    queue.write_json(OPS / 'expected_target_contract.json', target)
    queue.write_json(OPS / 'preflight.json', {'status': 'verified_read_only', 'at': queue.now(),
        'source': report, 'budget': measured, 'code_files': len(plan['code_sha256']),
        'runtime_identity_matches': True, 'plan_sha256': plan['_sha256']})
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = read(OPS / 'plan.json')
    plan['_sha256'] = queue.sha256(OPS / 'plan.json')
    if args.check:
        print(json.dumps(preflight(plan), indent=2))
        return
    with ExitStack() as stack:
        lock = stack.enter_context((OPS / 'controller.lock').open('a+'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OPS / 'status.json').exists():
            raise RuntimeError('Existing controller state; inspect before restart')
        state = {'experiment': plan['experiment'], 'controller_pid': os.getpid(),
            'controller_start_ticks': queue.process_info(os.getpid())['start_ticks'],
            'started_at': queue.now(), 'source_step': 200, 'resume_step': 200, 'target_step': 250,
            'plan_sha256': plan['_sha256'], 'sync_verification': 'pending', 'training_started': False}
        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(OPS / 'status.json', state)
            print(f'[{queue.now()}] {stage}: {json.dumps(details, ensure_ascii=False)}', flush=True)
        try:
            clean.acquire_resource_locks(stack, plan, status)
            measured = preflight(plan)
            if queue.gpu_pids():
                raise RuntimeError('GPUs occupied')
            if list((OPS / 'sync_receipts').glob('*.json')):
                raise RuntimeError('Refusing existing synchronization receipts')
            preserve_adapters(plan)
            env = clean.environment(plan, 'train')
            env.update(TARGET_STEP='250', RESUME_MODE='auto', RAY_NUM_CPUS='20',
                RAY_TEMP_BASE='/tmp/gt32_resume250_1016', RAY_DEDUP_LOGS='0',
                LOG_FILE=str(OPS / 'train_to250_metrics.log'),
                LORA_INIT_AUDIT_PATH=str(OPS / 'initialization.json'), **plan['sync_environment'])
            launch = ['bash', str(ROOT / 'scripts/local/run_rlvr_protocol_1p5b_4gpu.sh')]
            with (OPS / 'launcher_dry_run.log').open('x') as handle:
                subprocess.run(launch, env={**env, 'DRY_RUN': '1'}, cwd=ROOT,
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
            actual = read(plan['source_contract']['path'])
            cont.assert_contract_extension(read(OPS / 'expected_target_contract.json'), actual, 250, 250)
            queue.write_json(OPS / 'launched_contract.json', actual)
            guard(plan)
            log = OPS / 'train_to250.log'
            source = Path(plan['checkpoint_root']) / 'global_step_200'
            verified = False
            def completed():
                nonlocal verified
                cont.verify_resume_log(log, source)
                receipts = sync_receipts(plan)
                if receipts and not verified:
                    verified = True
                    queue.write_json(OPS / 'first_rollout_verification.json', receipts)
                    status('first_rollout_verified', sync_verification='verified',
                           verification=str(OPS / 'first_rollout_verification.json'))
                step = queue.recent_step(log)
                if step and step >= 201 and not verified:
                    raise RuntimeError('Update observed without first-rollout synchronization verification')
                if queue.training_complete(plan):
                    cont.verify_resume_log(log, source, required=True)
                    sync_receipts(plan, required=True)
                    return True
                return False
            status('starting_verified_continuation', training_started=True, resume_from=str(source))
            try:
                cont.run_continuation_stage(plan, 'train_to250', launch, env, completed,
                    status, lambda: guard(plan), measured)
            finally:
                restore_adapters()
            terminal = checkpoint(plan, 250)
            queue.write_json(OPS / 'terminal_checkpoint_verification.json', terminal)
            status('training_complete', latest_logged_step=250)
            adapter = Path(plan['checkpoint_root']) / 'global_step_250/actor/peft_adapter'
            clean.run_evaluation(plan, 250, adapter, status, workspace=OPS, guard=lambda: guard(plan))
            status('reporting', latest_logged_step=250)
            with (OPS / 'report.log').open('x') as handle:
                subprocess.run([plan['python'], str(OPS / 'report_results.py'), '--final'],
                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            guard(plan)
            status('complete', auto_resume=False, results=str(OPS / 'results.md'))
        except BaseException as error:
            status('error', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
