#!/usr/bin/env python3
"""User-directed stop at durable step210, then evaluate210/200/150."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path('/root/peft-for-rl')
OPS = Path(__file__).resolve().parent
OLD = ROOT / 'runs/ops/real_p256_r32_longrun270_20261017'
sys.path.insert(0, str(ROOT))
from scripts.local import run_real_after_eval as queue
from scripts.local import run_gradtop_clean_after_real as clean
from scripts.local.real_experiment_eval import _load_file, _check_adapter


def read(path):
    return json.loads(Path(path).read_text())


def guard(plan):
    queue.verify_code(plan)
    clean.assert_plan_unchanged(OPS / 'plan.json', plan['_sha256'])
    clean.assert_plan_unchanged(OLD / 'plan.json', plan['original_plan_sha256'])
    for path, digest in read(OLD / 'artifact_binding.json').items():
        clean.assert_plan_unchanged(path, digest)


def validate_snapshot(plan, step):
    from safetensors import safe_open
    directory = Path(plan['snapshot_root']) / f'step_{step:03d}'
    manifest = read(directory / 'manifest.json')
    if (manifest['trainer_step'], manifest['optimizer_update'], manifest['tensor_count'],
            manifest['module_count'], manifest['rank'], manifest['alpha']) != (step, step*4, 392, 196, 32, 64):
        raise RuntimeError('Snapshot metadata mismatch')
    path = directory / 'adapter_model.safetensors'
    with safe_open(str(path), framework='pt') as handle:
        if len(handle.keys()) != 392 or handle.metadata()['trainer_step'] != str(step):
            raise RuntimeError('Snapshot header mismatch')
    return {'step': step, 'optimizer_updates': step*4, 'path': str(path),
            'sha256': queue.sha256(path), 'manifest': manifest, 'resumable': False}


def stop_old_controller(plan):
    expected = plan['original_controller']
    current = queue.process_info(expected['pid'])
    if current and current['state'] != 'Z' and current['start_ticks'] == expected['start_ticks']:
        os.kill(expected['pid'], signal.SIGTERM)
        for _ in range(100):
            current = queue.process_info(expected['pid'])
            if not current or current['state'] == 'Z' or current['start_ticks'] != expected['start_ticks']:
                return
            time.sleep(.1)
        raise RuntimeError('Original controller failed to exit')


def main():
    plan = read(OPS / 'plan.json')
    plan['_sha256'] = queue.sha256(OPS / 'plan.json')
    state = {'experiment': plan['experiment'], 'controller_pid': os.getpid(),
             'controller_start_ticks': queue.process_info(os.getpid())['start_ticks'],
             'target_step': 210, 'evaluations': [210, 200, 150], 'started_at': queue.now(),
             'completed_evaluations': [], 'plan_sha256': plan['_sha256']}
    def status(stage, **details):
        state.update(status=stage, updated_at=queue.now(), **details)
        queue.write_json(OPS / 'status.json', state)
        queue.write_json(OLD / 'status.json', {**state, 'active_control_directory': str(OPS),
            'original_target_step': 270, 'user_overrode_target': True})
        print(f'[{queue.now()}] {stage}: {json.dumps(details)}', flush=True)
    with ExitStack() as stack:
        lock = stack.enter_context((OPS / 'controller.lock').open('a+'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OPS / 'status.json').exists():
            raise RuntimeError('Already registered; inspect before restarting')
        guard(plan)
        for step in (200, 150):
            validate_snapshot(plan, step)
        latest = queue.recent_step(OLD / 'train.log')
        if latest is None or latest > 210:
            raise RuntimeError(f'Cannot apply exact210 stop at current step {latest}')
        stop_old_controller(plan)
        for path in [OLD / 'controller.lock', *map(Path, plan['resource_locks'])]:
            handle = stack.enter_context(path.open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('waiting_for_step210', latest_logged_step=latest, stage_log=str(OLD / 'train.log'))
            heartbeat = time.monotonic()
            while True:
                latest = queue.recent_step(OLD / 'train.log')
                ready = Path(plan['snapshot_root']) / 'step_210/manifest.json'
                if latest and latest >= 210 and ready.is_file():
                    if latest != 210:
                        raise RuntimeError(f'Trainer advanced past210: {latest}')
                    # Freeze only this training stage before hashing its completed snapshot.
                    frozen = []
                    for proc in queue.owned_stage_processes(plan['training_stage_token']):
                        current = queue.process_info(proc['pid'])
                        if current and current['start_ticks'] == proc['start_ticks']:
                            try:
                                os.kill(proc['pid'], signal.SIGSTOP)
                                frozen.append(proc)
                            except ProcessLookupError:
                                pass
                    queue.write_json(OPS / 'frozen_training_processes.json', frozen)
                    snapshot = validate_snapshot(plan, 210)
                    queue.write_json(OPS / 'terminal_snapshot_verification.json', snapshot)
                    status('stopping_at210', latest_logged_step=210, terminal_adapter_verified=True)
                    # SIGKILL avoids allowing a stopped worker to perform another update.
                    for proc in frozen:
                        current = queue.process_info(proc['pid'])
                        if current and current['start_ticks'] == proc['start_ticks']:
                            try:
                                os.kill(proc['pid'], signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    queue.cleanup_stage(plan['training_stage_token'])
                    queue.wait_idle(status)
                    if queue.recent_step(OLD / 'train.log') != 210:
                        raise RuntimeError('Post-stop log has a different final completed step')
                    status('training_stopped_at210', latest_logged_step=210,
                           terminal_full_checkpoint=False, latest_resumable_step=200)
                    break
                if time.monotonic() - heartbeat > 30:
                    if not queue.owned_stage_processes(plan['training_stage_token']):
                        raise RuntimeError('Training exited before step210')
                    status('waiting_for_step210', latest_logged_step=latest)
                    heartbeat = time.monotonic()
                time.sleep(.1)
            guard(plan)
            clean.verify_eval_assets(plan)
            helper = _load_file('real210_original_helpers', OLD / 'controller.py')
            helper.OPS = OPS
            completed = []
            for step in plan['evaluations']:
                guard(plan)
                validate_snapshot(plan, step)
                adapter = helper.export_eval_adapter(plan, step)
                _check_adapter(adapter, plan['base_model'])
                clean.run_evaluation(plan, step, adapter, status, workspace=OPS, guard=lambda: guard(plan))
                completed.append(step)
                helper.write_results(plan, completed)
                # The original report's fixed270 title is amended to the actual stop boundary.
                report = OPS / 'results.md'
                report.write_text(report.read_text().replace('continuous training.',
                    'continuous training through210; evaluation order210/200/150.'))
                status('evaluation_complete', completed_evaluations=completed)
            status('complete', completed_evaluations=completed, results=str(OPS / 'results.md'), auto_resume=False)
        except BaseException as error:
            status('error', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
