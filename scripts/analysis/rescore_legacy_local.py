#!/usr/bin/env python3
"""Replay the original local fullbench judge on saved responses, with original gold."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tina_run'))
from tina.analysis.rollout_utils import evaluate_math_answer
from fullbench_statistics import summarize

VERSION = 'legacy_local_fullbench_v1'
ORDER = ['aime24', 'aime25', 'amc23', 'hmmt_feb', 'math500', 'minerva']
EXPECTED = {'aime24': 960, 'aime25': 960, 'amc23': 1280, 'hmmt_feb': 960, 'math500': 2000, 'minerva': 1088}


def score(row):
    answer, correct = evaluate_math_answer(row['completion'], row['gold_answer'])
    return {**row, 'source_judgment': {k: row.get(k) for k in ('correct', 'format_valid', 'extracted_answer', 'judge_version', 'judge_source_sha256', 'judge_status')},
            'correct': bool(correct), 'format_valid': answer is not None, 'extracted_answer': answer,
            'judge_version': VERSION, 'judge_status': 'legacy_extracted' if answer is not None else 'legacy_no_answer'}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    current = (ROOT / 'tina_run/tina/analysis/rollout_utils.py').read_bytes()
    expected_sha256 = '5f7483e0c330980bd115550dd2f8f11c759566c9dc1151a346721c9493d61d3f'
    if hashlib.sha256(current).hexdigest() != expected_sha256:
        raise RuntimeError('Local rollout_utils differs from the archived original; refusing a mixed judge')
    metadata = {'version': VERSION, 'reference_commit': '7ddb86a',
                'source_file': 'tina_run/tina/analysis/rollout_utils.py',
                'source_sha256': hashlib.sha256(current).hexdigest(),
                'entrypoint': 'evaluate_math_answer',
                'gold_policy': 'Original per-response gold_answer; no replacement, boxing, or normalization added',
                'dependencies': {name: version(name) for name in ['math-verify', 'latex2sympy2-extended', 'sympy']}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries, checks = [], []
    for source in args.records:
        raw = source.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if Counter(r['benchmark'] for r in rows) != EXPECTED or len({r['checkpoint'] for r in rows}) != 1:
            raise RuntimeError(f'Incomplete or mixed checkpoint records: {source}')
        print(f'Scoring {rows[0]["checkpoint"]}: {len(rows)} responses', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rescored = list(pool.map(score, rows, chunksize=16))
        summary = summarize(rescored)
        summary.update({'judge': metadata, 'source': str(source.resolve()), 'source_sha256': source_sha,
                        'source_judges': sorted({r.get('judge_version') or 'legacy_unversioned' for r in rows}),
                        'created_at_utc': datetime.now(timezone.utc).isoformat()})
        checkpoint = summary['checkpoint']
        for folder, selected in [('records', rescored), ('changes', [r for r in rescored if any(r[k] != r['source_judgment'][k] for k in ('correct', 'format_valid', 'extracted_answer'))])]:
            path = args.output_dir / folder / f'{checkpoint}.jsonl'
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix('.jsonl.tmp')
            with tmp.open('w') as handle:
                for row in selected:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            tmp.replace(path)
        atomic_json(args.output_dir / 'summary' / f'{checkpoint}.json', summary)
        # Re-read the output and independently check response identity, source immutability,
        # and exact replay of originally legacy-scored responses50/100.
        path = args.output_dir / 'records' / f'{checkpoint}.jsonl'
        decoded = [json.loads(line) for line in path.open()]
        invariant = ('checkpoint', 'benchmark', 'id', 'sample_index', 'gold_answer', 'prompt', 'completion',
                     'seed', 'temperature', 'top_p', 'max_new_tokens', 'max_model_len')
        preserved = len(decoded) == len(rows) and all(all(a[k] == b[k] for k in invariant) for a, b in zip(rows, decoded))
        changes = sum(a['correct'] != b['correct'] for a, b in zip(rows, decoded))
        legacy = not any('judge_version' in row for row in rows)
        if not preserved or hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError('Source responses or raw gold changed')
        check = {'checkpoint': checkpoint, 'status': 'verified', 'num_samples': len(decoded),
                 'original_records_unchanged': True, 'all_response_and_gold_fields_preserved': True,
                 'original_legacy_scored': legacy, 'changed_correctness_vs_input': changes,
                 'exact_replay_of_original_scores': (changes == 0) if legacy else None,
                 'source_sha256': source_sha,
                 'rescored_records_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if legacy and changes:
            raise RuntimeError(f'Original judge replay differs for {changes} responses; inspect changes before reporting')
        checks.append(check); summaries.append(summary)
        atomic_json(args.output_dir / 'verification.json', {'status': 'verified', 'judge': metadata, 'checkpoints': checks})
        print(json.dumps({'checkpoint': checkpoint, 'avg': summary['response_weighted_avg_at_k'], 'changes': changes}), flush=True)
    lines = ['原始本地评分：`legacy_local_fullbench_v1`，直接调用归档版本 `evaluate_math_answer`。',
             '使用原始 gold_answer；单元格为 Avg@k / Pass@k（%），Avg. 按回答数加权。', '',
             '| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |',
             '| ---- | ---- | --------- | --------- | ------ | ------- | --------- | --------- |']
    import csv
    compact = []
    for summary in summaries:
        tasks = {r['benchmark']: r for r in summary['benchmarks']}
        step = int(re.search(r'_step(\d+)_', summary['checkpoint'])[1])
        cells = [str(step), f'{100*summary["response_weighted_avg_at_k"]:.1f}']
        cells += [f'{100*tasks[b]["avg_at_k"]:.1f} / {100*tasks[b]["pass_at_k"]:.1f}' for b in ORDER]
        lines.append('| ' + ' | '.join(cells) + ' |')
        compact.append(dict(zip(['Step', 'Avg.', 'AIME24@32', 'AIME25@32', 'AMC@32', 'HMMT@32', 'MATH500@4', 'Minerva@4'], cells)))
    (args.output_dir / 's2_compact.md').write_text('\n'.join(lines) + '\n')
    with (args.output_dir / 's2_compact.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)


if __name__ == '__main__':
    main()
