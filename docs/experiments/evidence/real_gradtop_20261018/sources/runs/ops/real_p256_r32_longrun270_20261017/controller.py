#!/usr/bin/env python3
"""Fresh REAL-P256 -> continuous 270 steps -> six offline benchmark evaluations."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/peft-for-rl')
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.local import run_real_after_eval as queue
from scripts.local import run_gradtop_clean_after_real as clean
from scripts.local.real_experiment_eval import _load_file, _check_adapter
from scripts.local.run_real_cached_after_eval import weighted_summary, BENCHMARKS


def read(path):
    return json.loads(Path(path).read_text())


def guard(plan):
    clean.assert_plan_unchanged(OPS / 'plan.json', plan['_sha256'])
    queue.verify_code(plan)
    if (OPS / 'artifact_binding.json').exists():
        for path, digest in read(OPS / 'artifact_binding.json').items():
            clean.assert_plan_unchanged(path, digest)


def environment(plan):
    env = queue.common_environment(plan)
    allowed = {'PATH', 'CONDA_PREFIX', 'PYTHON_BIN', 'PYTHONPATH', 'PYTHONUNBUFFERED',
               'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'CUDA_VISIBLE_DEVICES', 'RUNTIME_ROOT',
               'DRY_RUN', 'RESUME_MODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'HOME', 'USER',
               'LOGNAME', 'SHELL', 'LANG', 'TMPDIR', 'TMP', 'TEMP', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH'}
    return {k: v for k, v in env.items() if k in allowed or k.startswith(('LC_', 'NCCL_'))}


def probe_environment(plan):
    return {**environment(plan), 'EXP_NAME': plan['probe_experiment'],
            'GRADIENT_PROBE_OUTPUT_DIR': plan['artifact_dir'],
            'REAL_PROBE_RANK': '32', 'REAL_PROBE_INPUT_WIDTH': '128', 'REAL_PROBE_OUTPUT_WIDTH': '128',
            'REAL_PROBE_STEPS': '4', 'REAL_PROBE_PROMPTS': '0', 'REAL_PROBE_SEED': '42',
            'TRAIN_PROMPT_BSZ': '64', 'TRAIN_PROMPT_MINI_BSZ': '16', 'N_RESP_PER_PROMPT': '8',
            'LR': '1e-5', 'LR_WARMUP_STEPS': '0', 'WEIGHT_DECAY': '0',
            'LORA_INIT_SEED': '42', 'DATA_SEED': '42', 'PPO_DATA_LOADER_SEED': '42',
            'RAY_TEMP_DIR': '/tmp/rp256_1017', 'RAY_NUM_CPUS': '20', 'RAY_DEDUP_LOGS': '0',
            'GPU_MEMORY_UTILIZATION': '0.70', 'ACTOR_PPO_MAX_TOKEN_LEN': '12288',
            'INFER_PPO_MAX_TOKEN_LEN': '12288', 'ROLLOUT_MAX_NUM_SEQS': '256',
            'ROLLOUT_MAX_NUM_BATCHED_TOKENS': '9216'}


def probe_complete(plan):
    artifact = Path(plan['artifact_dir'])
    if not all((artifact / n).is_file() for n in ('summary.json', 'rank_map.json', 'subspaces.safetensors')):
        return False
    s = read(artifact / 'summary.json')
    if (s['trainer_step'], s['optimizer_updates'], s['requested_observation_steps'],
        s['requested_prompt_groups']) != (4, 0, 4, 0) or not 0 < s['prompt_groups'] <= 256:
        raise RuntimeError('Probe exceeded or changed the registered budget')
    return True


def preflight(plan):
    guard(plan)
    queue.verify_assets(plan)
    clean.verify_eval_assets(plan)
    reference = read(ROOT / 'configs/experiments/gradtop_r32_clean_20261015.json')
    protocol = read(plan['protocol_file'])
    for key in ('assets', 'training', 'evaluation'):
        if reference[key] != protocol[key]:
            raise RuntimeError(f'Latest GradTop protocol mismatch: {key}')
    for key in ('artifact_dir', 'checkpoint_root', 'snapshot_root'):
        path = Path(plan[key])
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f'Refusing existing run data: {path}')
    if any(clean.evaluation_output(plan, s).exists() for s in plan['evaluations']):
        raise RuntimeError('Existing benchmark outputs')
    free = shutil.disk_usage(ROOT).free
    if free < plan['disk_budget']['required_free_bytes']:
        raise RuntimeError('Insufficient registered peak storage budget')
    if queue.gpu_pids():
        raise RuntimeError('GPU workers still active')
    with (OPS / 'probe_preflight.log').open('w') as handle:
        subprocess.run(['bash', str(ROOT / 'scripts/local/start_real_gradient_probe_4gpu.sh')],
                       env={**probe_environment(plan), 'DRY_RUN': '1'}, cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = {'status': 'verified', 'at': queue.now(), 'free_bytes': free,
              'disk_budget': plan['disk_budget'], 'matches_latest_gradtop_assets_and_training': True}
    queue.write_json(OPS / 'preflight.json', result)
    return result


def prepare_initial_source(plan):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(1)
    artifact = Path(plan['artifact_dir'])
    verifier = _load_file('realp256_protocol_check', ROOT / 'scripts/analysis/verify_rlvr_protocol.py')
    report = verifier.verify(argparse.Namespace(
        protocol=Path(plan['protocol_file']), learning_rate='1e-5', method='real', rank=32,
        target_step=270, real_artifact_dir=artifact, gradtop_artifact_dir=None,
        preview_missing_gradtop_artifact=False))
    queue.write_json(OPS / 'probe_artifact_verification.json', report)
    bases = load_file(str(artifact / 'subspaces.safetensors'))
    if len(bases) != 196:
        raise RuntimeError('REAL must cover 196 modules')
    tensors = {}
    with safe_open(str(Path(plan['base_model']) / 'model.safetensors'), framework='pt') as base:
        for name, basis in bases.items():
            shape = base.get_slice(name + '.weight').get_shape()
            if tuple(basis.shape) != (32, shape[1]) or not torch.isfinite(basis).all():
                raise RuntimeError('Invalid REAL input subspace')
            tensors[f'base_model.model.{name}.lora_A.weight'] = basis.float().contiguous()
            tensors[f'base_model.model.{name}.lora_B.weight'] = torch.zeros(shape[0], 32)
    directory = OPS / 'initial_adapter'
    directory.mkdir(exist_ok=False)
    source = directory / 'adapter_model.safetensors'
    save_file(tensors, str(source))
    from peft import LoraConfig
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0., bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                     rank_pattern={n: 32 for n in bases}, alpha_pattern={n: 64 for n in bases})
    cfg.base_model_name_or_path = plan['base_model']
    cfg.save_pretrained(directory)
    binding = {str(p): queue.sha256(p) for p in [
        artifact / 'summary.json', artifact / 'rank_map.json', artifact / 'subspaces.safetensors',
        source, directory / 'adapter_config.json']}
    queue.write_json(OPS / 'artifact_binding.json', binding)
    return source


def fresh_receipts(required=False):
    verifier = _load_file('realp256_sync_verifier', ROOT / 'verl/utils/lora_resume_verification.py')
    reports = {}
    for phase in ('initial', 'after_step1'):
        directory = OPS / 'sync_receipts' / phase
        report = verifier.validate_receipt_directory(directory, expected_world_size=4,
                                                     expected_checkpoint=directory, required=required)
        if report:
            for actor in report['receipts']['actor']:
                if actor.get('b_nonzero') != (phase == 'after_step1') or not actor.get('fixed_a_matches_probe'):
                    raise RuntimeError('Fresh initialization/update proof incomplete')
            reports[phase] = report
    return reports


def export_eval_adapter(plan, step):
    source = Path(plan['snapshot_root']) / f'step_{step:03d}'
    manifest = read(source / 'manifest.json')
    if (manifest['trainer_step'], manifest['optimizer_update'], manifest['module_count'],
        manifest['tensor_count'], manifest['rank'], manifest['alpha']) != (step, 4*step, 196, 392, 32, 64):
        raise RuntimeError('Snapshot provenance mismatch')
    adapter = OPS / 'evaluation_adapters' / f'step_{step:03d}'
    adapter.mkdir(parents=True, exist_ok=False)
    os.link(source / 'adapter_model.safetensors', adapter / 'adapter_model.safetensors')
    shutil.copy2(OPS / 'initial_adapter/adapter_config.json', adapter / 'adapter_config.json')
    _check_adapter(adapter, plan['base_model'])
    queue.write_json(adapter / 'export_provenance.json', {'source': str(source), 'manifest': manifest,
        'source_sha256': queue.sha256(source / 'adapter_model.safetensors'),
        'export_sha256': queue.sha256(adapter / 'adapter_model.safetensors')})
    return adapter


def checkpoint(plan):
    import torch
    torch.set_num_threads(1)
    if not queue.training_complete(plan):
        raise RuntimeError('Terminal checkpoint incomplete')
    updates = []
    for rank in range(4):
        source = Path(plan['checkpoint_root']) / f'global_step_270/actor/optim_world_size_4_rank_{rank}.pt'
        optim = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        values = {int(s['step']) for s in optim['state'].values()}
        if len(optim['state']) != 196 or values != {1080}:
            raise RuntimeError('REAL must have 196 trainable B factors and 1080 Adam updates')
        if any(float(g['lr']) != 1e-5 for g in optim['param_groups']):
            raise RuntimeError('Unexpected terminal optimizer learning rate')
        updates.append({'rank': rank, 'states': len(optim['state']), 'updates': sorted(values)})
    queue.write_json(OPS / 'terminal_checkpoint_verification.json', {'status': 'verified', 'optimizers': updates})


def write_results(plan, completed):
    rows = {str(s): weighted_summary(clean.evaluation_output(plan, s)) for s in completed}
    for row in rows.values():
        row['macro_avg'] = sum(b['avg_at_k'] for b in row['benchmarks'].values()) / 6
    queue.write_json(OPS / 'results.json', {'experiment': plan['experiment'], 'probe_attempted_prompts': 256,
        'accepted_prompt_groups': read(Path(plan['artifact_dir']) / 'summary.json')['prompt_groups'], 'steps': rows})
    lines = ['REAL-P256: rank32/alpha64, fixed A; fresh 256-prompt probe; continuous training.', '',
             'Cells are Avg@k / Pass@k (%). Avg. is response-weighted; Macro is the mean of six Avg@k.', '',
             '| Step | Avg. | Macro | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for step, row in rows.items():
        cells = [f"{100*row['benchmarks'][k]['avg_at_k']:.1f} / {100*row['benchmarks'][k]['pass_at_k']:.1f}" for k in BENCHMARKS]
        lines.append(f"| {step} | {100*row['weighted_avg']:.1f} | {100*row['macro_avg']:.2f} | " + ' | '.join(cells) + ' |')
    (OPS / 'results.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = read(OPS / 'plan.json')
    plan['_sha256'] = queue.sha256(OPS / 'plan.json')
    if args.check:
        print(json.dumps(preflight(plan)))
        return
    with ExitStack() as stack:
        for path in [str(OPS / 'controller.lock'), *plan['resource_locks']]:
            handle = stack.enter_context(Path(path).open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OPS / 'status.json').exists():
            raise RuntimeError('Existing controller status; inspect before restarting')
        state = {'experiment': plan['experiment'], 'controller_pid': os.getpid(),
                 'controller_start_ticks': queue.process_info(os.getpid())['start_ticks'],
                 'started_at': queue.now(), 'target_step': 270, 'plan_sha256': plan['_sha256']}
        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(OPS / 'status.json', state)
            print(f'[{queue.now()}] {stage}: {json.dumps(details, ensure_ascii=False)}', flush=True)
        try:
            preflight(plan)
            status('preflight_verified')
            queue.run_stage(plan, 'probe', ['bash', str(ROOT / 'scripts/local/start_real_gradient_probe_4gpu.sh')],
                            probe_environment(plan), lambda: probe_complete(plan), status)
            guard(plan)
            initial = prepare_initial_source(plan)
            status('probe_verified', attempted_prompts=256,
                   accepted_prompt_groups=read(Path(plan['artifact_dir']) / 'summary.json')['prompt_groups'])
            env = {**environment(plan), 'METHOD': 'real', 'REAL_ARTIFACT_DIR': plan['artifact_dir'],
                   'EXP_NAME': plan['experiment'], 'PROTOCOL_FILE': plan['protocol_file'],
                   'LR': '1e-5', 'LORA_RANK': '32', 'TARGET_STEP': '270',
                   'RAY_NUM_CPUS': '20', 'RAY_TEMP_DIR': '/tmp/rt256_1017', 'RAY_DEDUP_LOGS': '0',
                   'LOG_FILE': str(OPS / 'train_metrics.log'),
                   'VERL_LORA_FRESH_VERIFY_DIR': str(OPS / 'sync_receipts'),
                   'VERL_LORA_FRESH_VERIFY_INITIAL_SOURCE': str(initial),
                   'VERL_LORA_FRESH_VERIFY_INITIAL_SHA256': queue.sha256(initial),
                   'VERL_LORA_FRESH_VERIFY_WORLD_SIZE': '4', 'VERL_LORA_FRESH_VERIFY_TENSOR_COUNT': '392',
                   'VERL_LORA_FRESH_VERIFY_TOKEN': plan['sync_token']}
            launch = ['bash', str(ROOT / 'scripts/local/run_rlvr_protocol_1p5b_4gpu.sh')]
            with (OPS / 'training_preflight.log').open('x') as handle:
                subprocess.run(launch, env={**env, 'DRY_RUN': '1'}, cwd=ROOT,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            guard(plan)
            verified = set()
            def completed():
                reports = fresh_receipts()
                for phase, report in reports.items():
                    if phase not in verified:
                        queue.write_json(OPS / f'{phase}_sync_verification.json', report)
                        verified.add(phase)
                        status('fresh_rollout_verified', sync_phases_verified=sorted(verified))
                step = queue.recent_step(OPS / 'train.log')
                if step and step >= 2 and len(reports) != 2:
                    raise RuntimeError('Training advanced without both fresh synchronization gates')
                if queue.training_complete(plan):
                    fresh_receipts(required=True)
                    return True
                return False
            queue.run_stage(plan, 'train', launch, env, completed, status)
            checkpoint(plan)
            status('training_complete', latest_logged_step=270)
            completed_steps = []
            for step in plan['evaluations']:
                adapter = export_eval_adapter(plan, step)
                clean.run_evaluation(plan, step, adapter, status, workspace=OPS, guard=lambda: guard(plan))
                completed_steps.append(step)
                write_results(plan, completed_steps)
                status('evaluation_complete', completed_evaluations=completed_steps)
            status('complete', results=str(OPS / 'results.md'), auto_resume=False)
        except BaseException as error:
            status('error', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
