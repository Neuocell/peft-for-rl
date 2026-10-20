#!/usr/bin/env python3
"""Inventory saved local evaluations, rescore with PeRL, and retain historical scores."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ORDER = ['aime24', 'aime25', 'amc23', 'hmmt_feb', 'math500', 'minerva']
EXPECTED = {'aime24': (30, 32), 'aime25': (30, 32), 'amc23': (40, 32),
            'hmmt_feb': (30, 32), 'math500': (500, 4), 'minerva': (272, 4)}
DERIVED = ('evaluation_perl_release_judge', 'evaluation_rescore', 'evaluation_tina_',
           'evaluation_legacy_local', 'parser_fix_drafts', 'perl_release_judge')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def location(path):
    """Keep machine-readable/report provenance relative to the shared workspace."""
    return os.path.relpath(Path(path).resolve(), ROOT.parent)


def family(checkpoint):
    if checkpoint.startswith('base_'): return 'Base model (historical reference)'
    if checkpoint.startswith('gradtop_s2_'): return 'GradTop-S2'
    if checkpoint.startswith('gradtop_s3_'): return 'GradTop-S3'
    if 'syncfix2_refv2' in checkpoint: return 'NoRA refv2 / syncfix2'
    if 'nora_dora' in checkpoint:
        if 'from270' in checkpoint: return 'NoRA+DoRA syncfix1 / budget extension'
        if 'continuous' in checkpoint: return 'NoRA+DoRA continuous (historical)'
        if 'syncfix1' in checkpoint: return 'NoRA+DoRA syncfix1'
        return 'NoRA+DoRA v1 (historical)'
    if 'nora_doft' in checkpoint: return 'NoRA+DoFT (historical)'
    if 'nora_init' in checkpoint: return 'NoRA v1 (historical)'
    if 'dora_' in checkpoint: return 'DoRA syncfix1' if 'syncfix1' in checkpoint else 'DoRA v1'
    if 'full_parameter' in checkpoint: return 'Full parameter'
    if 'matched_standard_lora' in checkpoint: return 'Matched standard LoRA'
    if checkpoint.startswith('phase1_'): return 'Phase-1 (historical protocol)'
    if 'chord' in checkpoint: return 'GradTop extrapolation (historical)'
    if 'compact' in checkpoint: return 'GradTop compaction (historical)'
    if checkpoint.startswith('gradtop_'): return 'GradTop adaptive rank / lr1e-6 (historical)'
    if 'chr_rlpo' in checkpoint: return 'CHR/RLPO (historical)'
    if 'rlpo_init' in checkpoint: return 'RLPO (historical)'
    if 'heldout_rollout' in checkpoint: return 'DAPO heldout validation (not fullbench)'
    if 'geora' in checkpoint: return 'GeoRA (historical)'
    if 'spo_poetbs' in checkpoint: return 'SPO/POETBS (historical)'
    if 'biso' in checkpoint: return 'BiSO (historical)'
    if 'lora' in checkpoint: return 'LoRA (historical)'
    return 'Other historical'


def step(checkpoint):
    matches = re.findall(r'(?:^|_)step(\d+)(?:_|$)', checkpoint)
    return int(matches[-1]) if matches else None


def measure(rows):
    tasks = []
    groups = defaultdict(list)
    for row in rows: groups[row['benchmark']].append(row)
    for benchmark, group in sorted(groups.items()):
        problems = defaultdict(list)
        for row in group: problems[str(row['id'])].append(row)
        widths = {int(r['samples_per_problem']) for r in group}
        if len(widths) != 1 or any(len(p) != next(iter(widths)) for p in problems.values()):
            raise ValueError('Incomplete/unequal sampling: ' + benchmark)
        keys = {(str(r['id']), int(r['sample_index'])) for r in group}
        if len(keys) != len(group): raise ValueError('Duplicate samples: ' + benchmark)
        width = next(iter(widths))
        if any({int(r['sample_index']) for r in p} != set(range(width)) for p in problems.values()):
            raise ValueError('Missing/out-of-range sample indices: ' + benchmark)
        if any(len({(r['problem'], str(r['gold_answer'])) for r in p}) != 1 for p in problems.values()):
            raise ValueError('Inconsistent per-problem question/gold: ' + benchmark)
        tasks.append(dict(benchmark=benchmark, num_problems=len(problems), num_samples=len(group),
                          samples_per_problem=width, avg_at_k=sum(r['correct'] for r in group)/len(group),
                          pass_at_k=sum(any(r['correct'] for r in p) for p in problems.values())/len(problems)))
    return dict(num_samples=len(rows), response_weighted_avg_at_k=sum(r['correct'] for r in rows)/len(rows), benchmarks=tasks)


def inventory(roots, output, minerva_source):
    gold = {r['question']: str(r['answer']) for r in map(json.loads, minerva_source.read_text(encoding='utf-8').splitlines())}
    candidates, summaries, seen = [], [], set()
    for root in roots:
        for path in sorted(root.rglob('*.json')):
            if path.parent.name != 'summary' or output in path.parents or any(x in str(path) for x in DERIVED): continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(data, dict) and data.get('checkpoint') and isinstance(data.get('benchmarks'), list):
                    summaries.append(dict(checkpoint=data['checkpoint'], path=location(path),
                                          num_samples=data.get('num_samples'), judge=data.get('judge')))
            except (ValueError, OSError): continue
        for path in sorted(root.rglob('*.jsonl')):
            if path.parent.name != 'records' or output in path.parents or any(x in str(path) for x in DERIVED): continue
            resolved = path.resolve()
            if resolved in seen: continue
            seen.add(resolved)
            entry = dict(source=location(path), checkpoint=path.stem)
            try:
                before = path.stat()
                raw = path.read_bytes()
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if not rows: raise ValueError('Empty records')
                if any('benchmark' not in row for row in rows):
                    raise ValueError('Unsupported heldout/response schema: missing benchmark; preserve original validation separately')
                checkpoints = {r['checkpoint'] for r in rows}
                if len(checkpoints) != 1: raise ValueError('Mixed checkpoints')
                entry['checkpoint'] = rows[0]['checkpoint']
                entry['source_sha256'] = hashlib.sha256(raw).hexdigest()
                entry['source_size'] = len(raw)
                entry['source_judges'] = sorted({r.get('judge_version') or 'legacy_unversioned' for r in rows})
                entry['historical'] = measure(rows)
                entry['fullbench'] = {t['benchmark']: (t['num_problems'], t['samples_per_problem']) for t in entry['historical']['benchmarks']} == EXPECTED
                if any(r['problem'] not in gold for r in rows if r['benchmark'] == 'minerva'):
                    raise ValueError('Minerva question does not exactly match the pinned PeRL gold dataset')
                entry['minerva_exact_matched_questions'] = len({r['problem'] for r in rows if r['benchmark'] == 'minerva'})
                entry['generation_values'] = {key: sorted({json.dumps(r.get(key), ensure_ascii=False, sort_keys=True) for r in rows})
                                              for key in ['base_model', 'seed', 'temperature', 'top_p', 'max_new_tokens', 'max_model_len']}
                question_records = sorted({(r['benchmark'], str(r['id']), r['problem'], str(r['gold_answer'])) for r in rows})
                entry['question_gold_sha256'] = hashlib.sha256(json.dumps(question_records, ensure_ascii=False).encode()).hexdigest()
                prompt_records = sorted({(r['benchmark'], str(r['id']), r['prompt']) for r in rows})
                entry['prompts_sha256'] = hashlib.sha256(json.dumps(prompt_records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                entry['family'], entry['step'] = family(entry['checkpoint']), step(entry['checkpoint'])
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Source is still changing')
                entry['status'] = 'ready'
            except (ValueError, KeyError, OSError, TypeError) as exc:
                entry['status'], entry['reason'] = 'excluded', str(exc)
            candidates.append(entry)
            print('Inventory:', entry['status'], entry['checkpoint'], flush=True)
    # Same checkpoint aliases must have exactly the same original bytes.
    checkpoints = {}
    for entry in candidates:
        if entry['status'] != 'ready': continue
        previous = checkpoints.get(entry['checkpoint'])
        if previous:
            entry['status'] = 'alias' if previous['source_sha256'] == entry['source_sha256'] else 'excluded'
            entry['reason'] = 'Duplicate source alias' if entry['status'] == 'alias' else 'Conflicting sources for same checkpoint'
            if entry['status'] == 'excluded':
                previous['status'], previous['reason'] = 'excluded', entry['reason']
        else: checkpoints[entry['checkpoint']] = entry
    ready_names = {e['checkpoint'] for e in candidates if e['status'] == 'ready'}
    unique_summaries = {}
    for summary in summaries:
        if summary['checkpoint'] in ready_names: continue
        if summary['checkpoint'] in unique_summaries:
            unique_summaries[summary['checkpoint']].setdefault('aliases', []).append(summary['path'])
        else: unique_summaries[summary['checkpoint']] = summary
    unavailable = list(unique_summaries.values())
    historical_registry = ROOT/'runs/phase1-signal-random-v1/analysis/local_fullbench_evidence_inventory_20261004.json'
    if historical_registry.exists():
        registry = {e['evaluation_name']:e for e in json.loads(historical_registry.read_text(encoding='utf-8'))['entries']}
        for entry in candidates:
            if entry['checkpoint'] in registry:
                evidence = registry[entry['checkpoint']]
                entry['historical_evidence_grade'] = evidence['evidence_grade']
                entry['historical_permitted_use'] = evidence['permitted_use']
                entry['historical_grade_reasons'] = evidence['grade_reasons']
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), roots=[location(r) for r in roots],
                  candidates=candidates, summary_without_ready_records=unavailable)
    atomic_json(output / 'inventory.json', result)
    return result


def cells(summary):
    tasks = {t['benchmark']: t for t in summary['benchmarks']}
    return [f'{100*summary["response_weighted_avg_at_k"]:.1f}'] + [
        f'{100*tasks[b]["avg_at_k"]:.1f} / {100*tasks[b]["pass_at_k"]:.1f}' if b in tasks else '—' for b in ORDER]


def merge_inventories(inventories):
    candidates, summaries, roots = [], {}, set()
    seen_sources = set()
    for data in inventories:
        roots.update(data['roots'])
        for entry in data['candidates']:
            if entry['source'] not in seen_sources:
                candidates.append(entry); seen_sources.add(entry['source'])
        for summary in data['summary_without_ready_records']:
            if summary['checkpoint'] not in summaries: summaries[summary['checkpoint']] = summary
            else: summaries[summary['checkpoint']].setdefault('aliases', []).append(summary['path'])
    names = {}
    for entry in candidates:
        if entry['status'] != 'ready': continue
        previous = names.get(entry['checkpoint'])
        if previous:
            if previous['source_sha256'] != entry['source_sha256']:
                raise ValueError('Conflicting checkpoint across inventories: ' + entry['checkpoint'])
            entry['status'], entry['reason'] = 'alias', 'Duplicate source alias'
        else: names[entry['checkpoint']] = entry
    registry_path = ROOT/'runs/phase1-signal-random-v1/analysis/local_fullbench_evidence_inventory_20261004.json'
    if registry_path.exists():
        registry = {e['evaluation_name']:e for e in json.loads(registry_path.read_text(encoding='utf-8'))['entries']}
        for entry in candidates:
            evidence = registry.get(entry['checkpoint'])
            if evidence:
                entry.update(historical_evidence_grade=evidence['evidence_grade'],historical_permitted_use=evidence['permitted_use'],historical_grade_reasons=evidence['grade_reasons'])
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),roots=sorted(roots),candidates=candidates,
                summary_without_ready_records=[s for name,s in summaries.items() if name not in names])


def report(data, output, destination):
    completed, failed = [], []
    for entry in data['candidates']:
        if entry['status'] != 'ready': continue
        path = output / 'perl_release_judge/summary' / (entry['checkpoint'] + '.json')
        if not path.exists(): failed.append(entry); continue
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary['source_sha256'] != entry['source_sha256']: raise ValueError('Stale summary: ' + str(path))
        if summary['judge']['version'] != 'perl_release_judge_v1': raise ValueError('Mixed judge')
        item = {**entry, 'current': summary, 'current_summary': location(path)}
        # For later S2 generation the inline judge was v2; obtain actual legacy replay separately.
        if entry['source_judges'] == ['fullbench_math_v2']:
            source = ROOT.parent / entry['source']
            legacy_path = source.parents[2] / 'evaluation_legacy_local_replay/summary' / (entry['checkpoint'] + '.json')
            if legacy_path.exists():
                legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
                if legacy['source_sha256'] == entry['source_sha256']:
                    item['legacy_replay'] = legacy
        completed.append(item)
    completed.sort(key=lambda e:(e['family'], e['step'] or 0, e['checkpoint']))
    lines = ['# 本机保存实验的双协议结果报告', '',
             '生成时间（UTC）：' + datetime.now(timezone.utc).isoformat(), '',
             '当前口径为 `perl_release_judge_v1`；历史本地重放版本为 `legacy_local_fullbench_v1`。无版本标签的旧文件按“原保存判分”报告，不声称已逐条重放为 v1。定义和复现入口见 [双协议说明](../evaluation_protocols_zh.md)。', '',
             '本报告覆盖扫描根目录中可读取的历史评测，包含本机生成和迁入的记录；不把“本机可读”解释为全部在本机训练。重新判分保留每条原始生成、prompt、gold 和采样参数。', '',
             f'已完成 **{len(completed)}** 组回答的 PeRL 重判，其中 **{sum(e["fullbench"] for e in completed)}** 组为完整六任务 @32/@4；其余为 screen 或单任务，单列。', '',
             '同一 judge 不代表训练配置、题目快照或提示词相同。表格按轨迹分组，不跨组排序；lr、rank、seed、syncfix、resume 和预算差异仍由 checkpoint 名称及原实验记录约束。', '',
             'Avg. 按回答数加权；任务单元格为 Avg@k / Pass@k（%）。六任务标准题数为 30/30/40/30/500/272，总回答 7248。', '']
    verification_path = output/'verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        verified = {e['checkpoint']:e for e in verification['checkpoints']}
        if (set(verified)=={e['checkpoint'] for e in completed}
                and all(verified[e['checkpoint']]['source_sha256']==e['source_sha256'] for e in completed)):
            import os
            link = os.path.relpath(verification_path,destination.parent)
            lines += [f'独立逐条核验通过：**{verification["experiments"]}** 组、**{verification["responses"]:,}** 个回答。原始字段、源文件 SHA256、source_judgment 和 gold 映射全部保留/核准；Avg@k、Pass@k 与总体加权 Avg. 独立复算一致。[核验回执]({link})。', '']
    header = ['Step', 'Avg.', 'AIME24@32', 'AIME25@32', 'AMC@32', 'HMMT@32', 'MATH500@4', 'Minerva@4']
    def table(entries, current=True):
        result = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |']
        for e in entries:
            summary = e['current'] if current else e.get('legacy_replay',e['historical'])
            result.append('| ' + ' | '.join([str(e['step']) if e['step'] is not None else '—', *cells(summary)]) + ' |')
        return result
    for current, title in [(True, '当前 PeRL 口径：完整六任务'), (False, '历史保存口径与 v1 重放：完整六任务')]:
        lines += ['## ' + title, '']
        groups = defaultdict(list)
        for e in completed:
            if not e['fullbench']: continue
            if not current and e['source_judges'] != ['legacy_unversioned'] and 'legacy_replay' not in e: continue
            # Group by exact trajectory prefix to avoid ambiguous duplicate step rows.
            trajectory = re.sub(r'_step\d+(?=_|$)', '', e['checkpoint'])
            groups[(e['family'], trajectory)].append(e)
        for (label, trajectory), entries in groups.items():
            lines += ['### ' + label, '', '`' + trajectory + '`', '']
            evidence_groups = defaultdict(list)
            for e in entries:
                if e.get('historical_evidence_grade'):
                    evidence_groups[(e['historical_evidence_grade'], e['historical_permitted_use'])].append(str(e['step']) if e['step'] is not None else 'no-RL')
            for (grade, permitted), steps in evidence_groups.items():
                lines += [f'历史证据等级（step {", ".join(steps)}）：**{grade}**；原允许用途：{permitted}。重判不升级该等级。', '']
            lines += table(entries, current)
            lines += ['', '证据：', '']
            for e in entries:
                path = ROOT.parent / (e['current_summary'] if current else e['source'])
                import os
                target = os.path.relpath(path, destination.parent)
                step_label = str(e['step']) if e['step'] is not None else '—（no-RL）'
                lines += [f'- step {step_label}：[记录/汇总]({target})；checkpoint：`{e["checkpoint"]}`。']
            lines += ['']
    lines += ['## 非完整六任务的独立结果', '', '以下 Avg. 只针对实际任务/子集计算，不能与六任务总体 Avg. 比较。', '',
              '| Checkpoint | 回答数 | 当前 Avg. | 原保存 Avg. | 任务与 k |', '| --- | ---: | ---: | ---: | --- |']
    for e in completed:
        if e['fullbench']: continue
        tasks = ', '.join(f'{t["benchmark"]}@{t["samples_per_problem"]}({t["num_problems"]}题)' for t in e['current']['benchmarks'])
        lines += [f'| `{e["checkpoint"]}` | {e["current"]["num_samples"]} | {cells(e["current"])[0]} | {cells(e["historical"])[0]} | {tasks} |']
    lines += ['', '## 新旧口径变化与来源', '', '分数变化来自同一回答的判分规则变化，不是重新训练或重新生成带来的提升。`原保存 judge` 为 v2 的条目不冒充历史 v1；若存在经过哈希匹配的 v1 重放，另列其 Avg.。', '',
              '| Checkpoint | 原保存 judge | 原保存 Avg. | v1重放 Avg. | PeRL Avg. | 错→对 | 对→错 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for e in completed:
        s=e['current']; old=e['historical']
        lines += [f'| `{e["checkpoint"]}` | {", ".join(e["source_judges"])} | {cells(old)[0]} | {cells(e["legacy_replay"])[0] if "legacy_replay" in e else "—"} | {cells(s)[0]} | {sum(t["wrong_to_correct"] for t in s["benchmarks"])} | {sum(t["correct_to_wrong"] for t in s["benchmarks"])} |']
    lines += ['', '## 无法按当前口径重报告的历史汇总', '',
              '下列 checkpoint 有原汇总而没有通过本次核验的原始回答。保留旧报告及其数字，当前口径记为“无法复评”，不由旧分数推算新分数。部分条目属于其他机器迁入的归档。', '',
              '| Checkpoint | 原回答数 | 当前口径 | 原汇总位置 |', '| --- | ---: | --- | --- |']
    for e in data['summary_without_ready_records']:
        lines += [f'| `{e["checkpoint"]}` | {e["num_samples"]} | 无法复评 | `{e["path"]}` |']
    lines += ['', '## 排除与待完成记录', '', '| 来源 | 状态 | 原因 |', '| --- | --- | --- |']
    for e in data['candidates']:
        if e['status'] not in ['ready', 'alias']:
            reason = e.get('reason', '')
            if reason == "'benchmark'": reason = '训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测'
            lines += [f'| `{e["source"]}` | {e["status"]} | {reason} |']
    for e in failed: lines += [f'| `{e["source"]}` | 未完成重判 | 见 batch_status.json 与日志 |']
    lines += ['', '复现命令、源码版本及协议边界见双协议说明。机器可读清单、逐任务精确值、原文件 SHA256 和题目/prompt 指纹保存在同目录结果 CSV 与运行目录 inventory.json。', '']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines), encoding='utf-8')
    csv_path = destination.with_suffix('.csv')
    fields = ['checkpoint','family','step','fullbench','protocol','benchmark','num_problems','samples_per_problem','num_samples','avg_at_k','pass_at_k','response_weighted_avg_at_k','source','source_sha256','question_gold_sha256','prompts_sha256']
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for e in completed:
            variants=[('perl_release_judge_v1', e['current']), ('source:' + ','.join(e['source_judges']),e['historical'])]
            if 'legacy_replay' in e: variants.append(('legacy_local_fullbench_v1',e['legacy_replay']))
            for protocol,s in variants:
                for t in s['benchmarks']:
                    writer.writerow({**{k:e[k] for k in ['checkpoint','family','step','fullbench','source','source_sha256','question_gold_sha256','prompts_sha256']},
                                     'protocol':protocol, 'response_weighted_avg_at_k':s['response_weighted_avg_at_k'],
                                     **{k:t[k] for k in ['benchmark','num_problems','samples_per_problem','num_samples','avg_at_k','pass_at_k']}})
    atomic_json(output/'report_verification.json',dict(status='complete' if not failed else 'partial', completed=len(completed), fullbench=sum(e['fullbench'] for e in completed), pending=len(failed), summary_only=len(data['summary_without_ready_records']), report=location(destination)))
    print('Report:', destination, 'completed=',len(completed),'pending=',len(failed), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',required=True,help='Run directories; sources are never modified')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--minerva-source',type=Path,default=ROOT/'configs/evaluation/assets/minerva_mathai.jsonl')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--reuse-inventory',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Render existing summaries without invoking the scorer')
    parser.add_argument('--merge-inventory',type=Path,nargs='+',help='Merge separately scanned roots without rereading generations')
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    data=json.loads((output/'inventory.json').read_text(encoding='utf-8')) if args.reuse_inventory else inventory([p.resolve() for p in args.roots],output,args.minerva_source)
    if args.merge_inventory:
        data=merge_inventories([data]+[json.loads(p.read_text(encoding='utf-8')) for p in args.merge_inventory])
        atomic_json(output/'inventory.json',data)
    if args.inventory_only: return
    if args.report_only:
        report(data,output,args.report.resolve())
        return
    status=[]
    for entry in data['candidates']:
        if entry['status'] != 'ready':continue
        command=[sys.executable,str(ROOT/'scripts/analysis/rescore_perl_public.py'),'--records',str(ROOT.parent/entry['source']),
                 '--output-dir',str(output/'perl_release_judge'),'--minerva-source',str(args.minerva_source),'--workers',str(args.workers)]
        print('Rescore:',entry['checkpoint'],flush=True)
        result=subprocess.run(command,check=False)
        status.append(dict(checkpoint=entry['checkpoint'],exit_code=result.returncode))
        atomic_json(output/'batch_status.json',status)
    report(data,output,args.report.resolve())
    if any(e['exit_code'] for e in status): raise SystemExit(1)


if __name__=='__main__': main()
