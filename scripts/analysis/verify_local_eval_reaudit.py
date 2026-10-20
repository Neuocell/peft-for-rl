#!/usr/bin/env python3
"""Verify every original field and recompute aggregate counts after CPU rescoring."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGED = {'correct', 'format_valid', 'extracted_answer', 'judge_version', 'judge_source_sha256', 'judge_status'}


def verify_one(task):
    entry, output, gold = task
    name=entry['checkpoint']
    source=ROOT.parent/entry['source']
    destination=output/'perl_release_judge/records'/(name+'.jsonl')
    summary=json.loads((output/'perl_release_judge/summary'/(name+'.json')).read_text(encoding='utf-8'))
    digest=hashlib.sha256(); output_digest=hashlib.sha256()
    counts=Counter(); correct=Counter(); groups=defaultdict(list)
    def original_lines(handle):
        for line in handle:
            digest.update(line)
            if line.strip(): yield json.loads(line)
    def rescored_lines(handle):
        for line in handle:
            output_digest.update(line)
            if line.strip(): yield json.loads(line)
    with source.open('rb') as src, destination.open('rb') as dst:
        for old,new in zip_longest(original_lines(src),rescored_lines(dst)):
            assert old is not None and new is not None, 'Response count changed: '+name
            assert all(new.get(k)==v for k,v in old.items() if k not in CHANGED), 'Original field changed: '+name
            assert new['source_judgment']=={k:old.get(k) for k in ('correct','format_valid','extracted_answer','judge_version','judge_source_sha256','judge_status')}, 'Source score lost: '+name
            assert new['judge_version']=='perl_release_judge_v1', 'Mixed judge: '+name
            assert new['judge_source_sha256']==summary['judge']['source_sha256']
            expected=gold[old['problem']] if old['benchmark']=='minerva' else str(old['gold_answer'])
            assert new['judge_gold_answer']==expected, 'Wrong gold mapping: '+name
            assert isinstance(new['correct'],bool)
            counts[new['benchmark']]+=1;correct[new['benchmark']]+=new['correct']
            groups[(new['benchmark'],str(new['id']))].append(new['correct'])
    assert digest.hexdigest()==entry['source_sha256']==summary['source_sha256'], 'Source hash changed: '+name
    assert sum(counts.values())==summary['num_samples']
    assert abs(sum(correct.values())/sum(counts.values())-summary['response_weighted_avg_at_k'])<1e-15
    for task in summary['benchmarks']:
        b=task['benchmark'];ps=[v for (bench,_),v in groups.items() if bench==b]
        assert len(ps)==task['num_problems'] and counts[b]==task['num_samples']
        assert abs(correct[b]/counts[b]-task['avg_at_k'])<1e-15
        assert abs(sum(any(p) for p in ps)/len(ps)-task['pass_at_k'])<1e-15
    return dict(checkpoint=name,status='verified',responses=sum(counts.values()),
                source_sha256=digest.hexdigest(),rescored_sha256=output_digest.hexdigest(),
                original_fields_preserved=True,source_judgment_preserved=True,gold_mapping_verified=True,
                independent_avg_pass_aggregation_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--minerva-source',type=Path,default=ROOT/'configs/evaluation/assets/minerva_mathai.jsonl')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();output=args.output_dir.resolve()
    data=json.loads((output/'inventory.json').read_text(encoding='utf-8'))
    gold={r['question']:str(r['answer']) for r in map(json.loads,args.minerva_source.read_text(encoding='utf-8').splitlines())}
    tasks=[(e,output,gold) for e in data['candidates'] if e['status']=='ready']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for result in pool.map(verify_one,tasks,chunksize=1):
            results.append(result)
            print('Verified:',result['checkpoint'],flush=True)
    result=dict(status='verified',experiments=len(results),responses=sum(r['responses'] for r in results),checkpoints=results)
    target=output/'verification.json';temp=target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(target)
    print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'}),flush=True)


if __name__=='__main__':main()
