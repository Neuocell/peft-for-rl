"""Exercise portable saved-response repair without generating new responses."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/analysis/rescore_perl_public.py'
GOLD = ROOT / 'configs/evaluation/assets/minerva_mathai.jsonl'


def test_cli_preserves_source_and_maps_minerva_gold(tmp_path):
    minerva = json.loads(GOLD.read_text(encoding='utf-8').splitlines()[0])
    rows = []
    for index, answer in enumerate(['42', '43']):
        rows.append(dict(checkpoint='portable_smoke', benchmark='aime24', id='q',
                         sample_index=index, samples_per_problem=2, problem='Question',
                         prompt='Original prompt', completion=r'\boxed{' + answer + '}',
                         gold_answer='42', correct=index == 1, format_valid=True,
                         extracted_answer=answer, completion_length=3, hit_max_new_tokens=False))
    rows.append({**rows[0], 'benchmark':'minerva', 'id':'m', 'sample_index':0,
                 'samples_per_problem':1, 'problem':minerva['question'],
                 'gold_answer':'Preserved legacy full solution',
                 'completion':r'\boxed{' + str(minerva['answer']) + '}'})
    source = tmp_path / 'original.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    before = source.read_bytes()
    output = tmp_path / 'perl'
    command = [sys.executable, str(SCRIPT), '--records', str(source),
               '--output-dir', str(output), '--workers', '1']
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert source.read_bytes() == before
    result = [json.loads(line) for line in (output/'records/portable_smoke.jsonl').read_text(encoding='utf-8').splitlines()]
    assert result[0]['correct'] is True and result[1]['correct'] is False
    assert result[2]['gold_answer'] == 'Preserved legacy full solution'
    assert result[2]['judge_gold_answer'] == str(minerva['answer'])
    assert all(r['completion'] == old['completion'] and r['prompt'] == old['prompt'] for r,old in zip(result,rows))
    summary = json.loads((output/'summary/portable_smoke.json').read_text(encoding='utf-8'))
    assert summary['source_sha256'] == hashlib.sha256(before).hexdigest()
    assert summary['minerva_exact_matched_questions'] == 1
    replay = subprocess.run(command, check=True, capture_output=True, text=True)
    assert 'Already verified' in replay.stdout


def test_cli_rejects_modified_gold_snapshot(tmp_path):
    modified = tmp_path/'minerva.jsonl'
    modified.write_bytes(GOLD.read_bytes() + b'\n')
    source = tmp_path/'unused.jsonl'
    source.write_text('{}\n', encoding='utf-8')
    result = subprocess.run([sys.executable, str(SCRIPT), '--records', str(source),
                             '--output-dir', str(tmp_path/'out'), '--minerva-source', str(modified)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Minerva gold differs' in result.stderr
