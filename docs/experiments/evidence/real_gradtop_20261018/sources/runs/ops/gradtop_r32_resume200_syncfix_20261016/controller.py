#!/usr/bin/env python3
"""Resume current GradTop with an explicit synchronization-verification boundary."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/peft-for-rl')
sys.path.insert(0, str(ROOT))
from scripts.local import continue_gradtop_r32 as cont
from scripts.local import run_gradtop_clean_after_real as clean
from scripts.local import run_real_after_eval as queue
from scripts.local.real_experiment_eval import _load_file

OPS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def guard(plan):
    clean.assert_plan_unchanged(OPS / 'plan.json', plan['_sha256'])
    queue.verify_code(plan)
    for path, expected in plan['source_artifact_sha256'].items():
        clean.assert_plan_unchanged(path, expected)
    for path, expected in plan['resume_source_sha256'].items():
        clean.assert_plan_unchanged(path, expected)


def budget(plan):
    ckpts = Path(plan['checkpoint_root'])
    source = ckpts / 'global_step_150'
    checkpoint_bytes = cont.directory_bytes(source)
    snapshot_bytes = cont.directory_bytes(Path(plan['snapshot_root']) / 'step_150')
    reclaim = 0
    for step in (50, 100):
        actor = ckpts / f'global_step_{step}' / 'actor'
        for p in actor.rglob('*'):
            if 'peft_adapter' in p.relative_to(actor).parts:
                continue  # These bytes are retained even before links exist.
            if p.is_file() and not p.is_symlink():
                st = p.stat()
                if st.st_dev == ckpts.stat().st_dev and st.st_nlink == 1:
                    reclaim += st.st_blocks * 512
    required = (math.ceil(((plan['target_step']-150)//10)*snapshot_bytes*1.10)
                +2*1024**3+max(0,math.ceil(checkpoint_bytes*1.01)-reclaim))
    return {'checkpoint_bytes':checkpoint_bytes,'snapshot_bytes':snapshot_bytes,
            'verified_retention_credit_bytes':reclaim,'required_free_bytes':required,
            'available_free_bytes':shutil.disk_usage(ROOT).free,
            'checkpoint_growth_margin':1.01,'snapshot_growth_margin':1.10,'headroom_gib':2,
            'retention':'pinned trainer deletes old50/100 actors before terminal save; original150 preserved'}


def preserve_adapters(plan):
    for step in (50,100,150):
        source=Path(plan['checkpoint_root'])/f'global_step_{step}'/'actor/peft_adapter'
        target=OPS/'preserved_adapters'/f'step_{step}'
        target.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if p.is_file():
                dest=target/p.name
                if not dest.exists():
                    os.link(p,dest,follow_symlinks=True)
                if queue.sha256(p)!=queue.sha256(dest):
                    raise RuntimeError('Preserved adapter content differs')


def restore_adapters(plan):
    for step in (50,100):
        source=OPS/'preserved_adapters'/f'step_{step}'
        target=Path(plan['checkpoint_root'])/f'global_step_{step}'/'actor/peft_adapter'
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(source,target_is_directory=True)


def runtime_migration(plan):
    tool=_load_file('gradtop_fixed_runtime',ROOT/'scripts/analysis/record_rlvr_runtime.py')
    old=read(plan['source_contract']['path'])
    clean.assert_plan_unchanged(plan['source_contract']['path'],plan['source_contract']['sha256'])
    new=cont.protocol_report(plan,plan['target_step'])
    cont.assert_contract_extension(old,new,150,plan['target_step'])
    identity=tool.make_identity(old,read(plan['protocol_file']),ROOT)
    if identity != tool.make_identity(new,read(plan['protocol_file']),ROOT):
        raise RuntimeError('Runtime identity changes beyond registered continuation boundary')
    manifest=ROOT/'runs/protocol_contracts'/f"{plan['experiment']}.runtime.json"
    record={'schema_version':1,'record_type':'verified_legacy_migration_not_clean_retraining',
            'created_at':queue.now(),'trajectory_origin':'legacy_gradtop_step150_with_known_step51_101_rollout_sync_defect',
            'identity':identity,'source_checkpoint':str(Path(plan['checkpoint_root'])/'global_step_150'),
            'source_contract_sha256':plan['source_contract']['sha256'],
            'source_plan':plan['source_plan'],'source_plan_sha256':plan['source_plan_sha256'],
            'first_fixed_resume_step':151,'source_sha256':plan['code_sha256'],
            'resume_plan':str(OPS/'plan.json'),'resume_plan_sha256':plan['_sha256']}
    if manifest.exists():
        existing=read(manifest)
        if existing.get('identity')!=identity or existing.get('source_contract_sha256')!=record['source_contract_sha256']:
            raise RuntimeError('Conflicting runtime migration already exists')
    else:
        with manifest.open('x') as f:
            json.dump(record,f,ensure_ascii=False,indent=2)
            f.write('\n')
    queue.write_json(OPS/'runtime_migration.json',record)
    queue.write_json(OPS/'contract_step150_original.json',old)
    queue.write_json(OPS/'expected_target_contract.json',new)


def sync_receipts(plan,required=False):
    # The verifier supplies its receipt validator; missing receipts are never
    # interpreted as successful synchronization.
    module=_load_file('gradtop_native_resume_verifier',ROOT/'verl/utils/lora_resume_verification.py')
    report = module.validate_receipt_directory(
        OPS/'sync_receipts',expected_world_size=4,
        expected_checkpoint=Path(plan['checkpoint_root'])/'global_step_150/actor',
        required=required,
    )
    if report:
        settings=plan['sync_environment']
        for field, variable in [('token','VERL_LORA_RESUME_VERIFY_TOKEN'),
                                ('source_file_sha256','VERL_LORA_RESUME_VERIFY_SHA256')]:
            if report[field] != settings[variable]:
                raise RuntimeError('Synchronization evidence belongs to another launch')
        if (report['tensor_count'] != 392 or
                Path(report['source']).resolve()!=Path(settings['VERL_LORA_RESUME_VERIFY_SOURCE']).resolve()):
            raise RuntimeError('Synchronization evidence has the wrong source/tensor count')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    plan=read(OPS/'plan.json')
    plan['_sha256']=queue.sha256(OPS/'plan.json')
    guard(plan)
    for path, expected in plan['source_checkpoint_sha256'].items():
        clean.assert_plan_unchanged(path,expected)
    cont.ensure_latest(plan,150)
    report=clean.validate_complete_checkpoint(plan,150)
    queue.write_json(OPS/'source_checkpoint150_verification.json',report)
    old=read(plan['source_contract']['path'])
    clean.assert_plan_unchanged(plan['source_contract']['path'],plan['source_contract']['sha256'])
    cont.assert_contract_extension(old,cont.protocol_report(plan,plan['target_step']),150,plan['target_step'])
    if args.check:
        queue.write_json(OPS/'preflight.json',{'status':'verified_read_only','at':queue.now(),
            'target_step':plan['target_step'],'plan_sha256':plan['_sha256'],'checkpoint':report,'disk_budget_before_preservation':budget(plan)})
        return
    with ExitStack() as stack:
        lock=stack.enter_context((OPS/'controller.lock').open('a+'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OPS/'status.json').exists():
            raise RuntimeError('Existing continuation state; inspect before recovery')
        state={'experiment':plan['experiment'],'controller_pid':os.getpid(),
               'controller_start_ticks':queue.process_info(os.getpid())['start_ticks'],
               'started_at':queue.now(),'target_step':plan['target_step'],'resume_step':150,
               'plan_sha256':plan['_sha256'],'sync_verification':'pending','training_started':False}
        def status(stage,**details):
            state.update(status=stage,updated_at=queue.now(),**details)
            queue.write_json(OPS/'status.json',state)
            print(f'[{queue.now()}] {stage}: {json.dumps(details,ensure_ascii=False)}',flush=True)
        try:
            dependency=plan['evaluation_dependency']
            while clean.process_alive(dependency['process']):
                guard(plan)
                status('waiting_for_current_evaluation',dependency=dependency['status'])
                time.sleep(20)
            prior=read(dependency['status'])
            if 110 not in prior.get('completed_steps',[]):
                raise RuntimeError('Current step110 evaluation did not finish verification')
            clean.acquire_resource_locks(stack,plan,status)
            guard(plan)
            if queue.gpu_pids():
                raise RuntimeError('GPUs remain occupied after evaluation handoff')
            if list((OPS/'sync_receipts').glob('*.json')):
                raise RuntimeError('Refusing to reuse pre-existing synchronization receipts')
            cutoff=plan.get('latest_training_start')
            if cutoff and datetime.now().astimezone()>datetime.fromisoformat(cutoff):
                status('not_started_insufficient_time',reason='Preserve prior17:00budget; safe terminal checkpoint cannot fit')
                return
            cont.ensure_latest(plan,150)
            preserve_adapters(plan)
            measured=budget(plan)
            queue.write_json(OPS/'disk_budget.json',measured)
            if measured['available_free_bytes']<measured['required_free_bytes']:
                raise RuntimeError('Insufficient measured disk budget')
            runtime_migration(plan)
            env=clean.environment(plan,'train')
            env.update(TARGET_STEP=str(plan['target_step']),RESUME_MODE='auto',RAY_NUM_CPUS='20',
                       RAY_TEMP_BASE='/tmp/gt32_syncfix_1016',RAY_DEDUP_LOGS='0',
                       LOG_FILE=str(OPS/f"train_to{plan['target_step']}_metrics.log"),
                       LORA_INIT_AUDIT_PATH=str(OPS/'initialization.json'),
                       **plan['sync_environment'])
            launch=['bash',str(ROOT/'scripts/local/run_rlvr_protocol_1p5b_4gpu.sh')]
            with (OPS/'launcher_dry_run.log').open('x') as f:
                subprocess.run(launch,env={**env,'DRY_RUN':'1'},cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            actual=read(plan['source_contract']['path'])
            cont.assert_contract_extension(read(OPS/'expected_target_contract.json'),actual,plan['target_step'],plan['target_step'])
            queue.write_json(OPS/'launched_contract.json',actual)
            log=OPS/f"train_to{plan['target_step']}.log"
            source=Path(plan['checkpoint_root'])/'global_step_150'
            first_verified=False
            def completed():
                nonlocal first_verified
                cont.verify_resume_log(log,source)
                receipts=sync_receipts(plan)
                if receipts and not first_verified:
                    first_verified=True
                    queue.write_json(OPS/'first_rollout_verification.json',receipts)
                    status('first_rollout_verified',sync_verification='verified',verification=str(OPS/'first_rollout_verification.json'))
                step=queue.recent_step(log)
                if step and step>=151 and not first_verified:
                    raise RuntimeError('Optimizer update observed without complete first-rollout verification receipts')
                if queue.training_complete(plan):
                    cont.verify_resume_log(log,source,required=True)
                    sync_receipts(plan,required=True)
                    return True
                return False
            status('starting_verified_continuation',training_started=True,resume_from=str(source))
            try:
                cont.run_continuation_stage(plan,f"train_to{plan['target_step']}",launch,env,completed,status,lambda:guard(plan),measured)
            finally:
                restore_adapters(plan)
            terminal=clean.validate_complete_checkpoint(plan,plan['target_step'])
            queue.write_json(OPS/'terminal_checkpoint_verification.json',terminal)
            status('training_complete',latest_logged_step=plan['target_step'],sync_verification='verified')
            if plan.get('evaluate_terminal'):
                adapter=Path(plan['checkpoint_root'])/f"global_step_{plan['target_step']}"/'actor/peft_adapter'
                clean.run_evaluation(plan,plan['target_step'],adapter,status,workspace=OPS,guard=lambda:guard(plan))
            status('complete',auto_resume=False)
        except BaseException as error:
            status('error',error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':
    main()
