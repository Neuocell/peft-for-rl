#!/usr/bin/env python3
"""Fixed-policy, paired-rollout gradient diagnostics; never updates policy weights."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OPS = ROOT / 'runs/ops/gradtop_real_gradient_diagnostics_20261017'
PYTHON = '/home/node/anaconda3/envs/peft-for-rl/bin/python'
MODULES = [f'model.layers.{layer}.{suffix}' for layer in (0, 13, 27)
           for suffix in ('self_attn.q_proj', 'mlp.down_proj')]

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    # JSON strings can legally contain U+2028; str.splitlines() splits those
    # inside one record. JSONL boundaries are physical newline-delimited lines.
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (42, *parts))).encode()).digest()[:4], 'big') % (2**31-1)

def register():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    protocol = read(ROOT / 'configs/experiments/gradtop_r32_clean_20261015.json')
    assets = protocol['assets']
    for key in ('base_model_weights', 'source_data', 'sampler_prefix_indices'):
        assert sha(ROOT / assets[key]['path']) == assets[key]['sha256'], key
    base = (ROOT / assets['base_model_weights']['path']).parent
    data = pq.read_table(ROOT / assets['source_data']['path']).to_pylist()
    indices = [int(x['source_row_index']) for x in csv.DictReader((ROOT/assets['sampler_prefix_indices']['path']).open())]
    excluded = set(indices[:4096])
    positions = {row: i+1 for i,row in enumerate(indices)}
    choices = sorted((i for i in range(len(data)) if i not in excluded),
                     key=lambda i: hashlib.sha256(('diagnostic42:'+data[i]['extra_info']['index']).encode()).hexdigest())[:64]
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    prompts = []
    for i,rowid in enumerate(choices):
        row = data[rowid]
        ids = tok.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
        # Same left truncation / chat template as production RLHFDataset.
        ids = ids[-1024:]
        content = row['prompt'][0]['content'].lower()
        tags = []
        for tag, words in [('geometry',['triangle','circle','angle','polygon']),
                           ('number_theory',['prime','divisible','remainder','divisor']),
                           ('combinatorics',['permutation','probability','ways','arrangements'])]:
            if any(w in content for w in words): tags.append(tag)
        prompts.append({'prompt_index': i, 'source_row': rowid, 'prompt_id': row['extra_info']['index'],
            'messages': row['prompt'], 'prompt_token_ids': ids, 'gold': row['reward_model']['ground_truth'],
            'data_source': row['data_source'], 'source_ability': row['ability'],
            'text_tag_heuristic': tags[0] if len(tags)==1 else 'unclassified',
            'training_position_1based': positions.get(rowid),
            'training_seen_by_step': {str(s): positions.get(rowid,10**9)<=64*s for s in (100,150,270)}})
    OPS.mkdir(exist_ok=True)
    write(OPS/'prompts.json',prompts)
    specs = [('gradtop100','gradtop',100), ('real100','real',100),
             ('gradtop150','gradtop',150), ('real150','real',150), ('gradtop270','gradtop',270)]
    paths = {'gradtop':'rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015',
             'real':'rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42'}
    policies = []
    for key,method,step in specs:
        adapter = ROOT/'runs/ckpts/verl/peft-for-rl'/paths[method]/f'global_step_{step}/actor/peft_adapter'
        policies.append({'key':key,'method':method,'step':step,'adapter':str(adapter),
                         'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
                         'adapter_config_sha256':sha(adapter/'adapter_config.json')})
    subspaces = {'gradtop_init':str(ROOT/'runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015/subspaces.safetensors'),
                 'real_init':str(ROOT/'runs/analysis/real_probe_r32_s32_seed42_replayed/subspaces.safetensors')}
    plan = {'schema_version':1,'registered_at':now(),'python':PYTHON,'base_model':str(base),
        'base_model_sha256':assets['base_model_weights']['sha256'], 'policies':policies,'modules':MODULES,
        'subspaces':subspaces,'subspace_sha256':{k:sha(v) for k,v in subspaces.items()},
        'prompts_sha256':sha(OPS/'prompts.json'),'prompt_count':64,'groups_per_prompt':2,'responses_per_group':8,
        'total_responses':5120,'max_response_tokens':8192,'max_prompt_tokens':1024,
        'temperature':1.0,'top_p':1.0,'top_k':-1,'output_projection_width':64,
        'projection_halves':{'A':32,'B':32},'formal_world_size':4,'formal_local_minibatch_responses':32,
        'formal_max_tokens':12288,'formal_updates_per_rollout':4,'optimizer_updates':0,
        'real270':'unavailable_not_synthesized',
        'sampling_exclusion':'first4096 production sampler rows, covering both original probes',
        'main_target':'fixed-policy on-policy clipped-PPO gradient at ratio=1; actual dynamic microbatch reduction replay',
        'branches':['equal_prompt_token_mean','global_token_mean16','formal_dynamic_replay','real_contrast_sensitivity_first32'],
        'formal_caveat':'fixed policy for all four updates; does not reconstruct actual historical Adam inputs',
        'raw_retention':'keep equal-prompt sketches; REAL first32 and formal partial sketches reduce to full paired Gram matrices then delete temporary shards; all response tokens and coefficients retained',
        'resource_locks':read(ROOT/'runs/ops/gradtop_r32_resume270_syncverified_20261017/plan.json')['resource_locks'] +
                         [str(ROOT/'runs/ops/gradtop_r32_resume270_syncverified_20261017/controller.lock')],
        'source_sha256':{str(ROOT/p):sha(ROOT/p) for p in [
            'scripts/analysis/run_gradtop_real_diagnostics.py',
            'scripts/analysis/summarize_gradtop_real_diagnostics.py',
            'verl/utils/seqlen_balancing.py','verl/workers/actor/dp_actor.py',
            'verl/trainer/ppo/core_algos.py','verl/trainer/ppo/ray_trainer.py',
            'verl/utils/lora_resume_verification.py','verl/utils/reward_score/boxed_math_accuracy.py']}}
    write(OPS/'plan.json',plan)
    print(json.dumps({'status':'registered','policies':len(policies),'responses':5120,'modules':MODULES},ensure_ascii=False))

def load_plan():
    p=read(OPS/'plan.json')
    assert sha(OPS/'prompts.json')==p['prompts_sha256']
    for path,h in p['source_sha256'].items():
        assert sha(path)==h, f'Frozen source changed: {path}'
    for k,path in p['subspaces'].items():
        assert sha(path)==p['subspace_sha256'][k],k
    return p

def policy(plan,key):
    p=next(p for p in plan['policies'] if p['key']==key)
    assert sha(Path(p['adapter'])/'adapter_model.safetensors')==p['adapter_sha256']
    assert sha(Path(p['adapter'])/'adapter_config.json')==p['adapter_config_sha256']
    return p

def engine_load_verify(worker, adapter, expected_sha, receipt_dir, token):
    """Run inside vLLM worker. Explicit load, activation and exact CUDA readback."""
    from safetensors.torch import load_file
    from vllm.lora.request import LoRARequest
    from verl.utils import lora_resume_verification as gate
    adapter=Path(adapter)
    env={'SOURCE':str(adapter/'adapter_model.safetensors'),'SHA256':expected_sha,
         'OUTPUT_DIR':receipt_dir,'TOKEN':token,'WORLD_SIZE':'1','TENSOR_COUNT':'392'}
    for k,v in env.items(): os.environ['VERL_LORA_RESUME_VERIFY_'+k]=v
    source=gate.canonical_tensors(load_file(str(adapter/'adapter_model.safetensors')))
    reference={'token':token,'source_file_sha256':expected_sha,'checkpoint_source':str(adapter.parent),
               'actor_payload_sha256':gate.tensor_fingerprint(source),
               'actor_dtypes':{k:str(v.dtype).removeprefix('torch.') for k,v in source.items()}}
    worker.add_lora(LoRARequest('diagnostic',123,str(adapter)))
    result=gate.verify_engine_adapter(worker,read(adapter/'adapter_config.json'),reference,123)
    # This receipt is a source->engine check, not a claim of an actor export.
    return {k:v for k,v in result.items() if k!='actor_dtypes'}

def generate(plan,key,shard,smoke=False):
    # Local, trusted callable RPC is required for source-to-CUDA adapter verification.
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from verl.utils.reward_score.boxed_math_accuracy import compute_score
    p=policy(plan,key); out=OPS/('smoke' if smoke else key);out.mkdir(exist_ok=True)
    prompts=read(OPS/'prompts.json')
    chosen=prompts[:1] if smoke else prompts[shard*16:(shard+1)*16]
    target=out/f'rollouts.{shard}.jsonl'
    if target.exists(): raise FileExistsError(target)
    llm=LLM(model=plan['base_model'],dtype='bfloat16',seed=seed_for(key,shard),
        max_model_len=9216,max_num_batched_tokens=9216,max_num_seqs=128,
        gpu_memory_utilization=0.80,tensor_parallel_size=1,enable_lora=True,max_loras=1,
        max_lora_rank=32,enable_prefix_caching=True,enforce_eager=True)
    import functools
    receipt=llm.collective_rpc(functools.partial(engine_load_verify,adapter=p['adapter'],
        expected_sha=p['adapter_sha256'],receipt_dir=str(out/f'engine_receipts.{shard}'),
        token=hashlib.sha256((p['adapter_sha256']+str(shard)).encode()).hexdigest()))
    assert len(receipt)==1 and receipt[0]['status']=='verified'
    write(out/f'engine_verification.{shard}.json',{'status':'verified','at':now(),'policy':key,'receipt':receipt,
         'request_adapter_id':123,'source_to_engine_only':True})
    start=time.monotonic()
    # A single batched call; independent request RNG seeds include repeat and stable prompt id.
    jobs=[(prompt,repeat) for prompt in chosen for repeat in range(2)]
    sampling=[SamplingParams(n=8,temperature=1.,top_p=1.,top_k=-1,
         max_tokens=128 if smoke else 8192,seed=seed_for('rollout',key,pr['prompt_id'],rep),
         detokenize=True,logprobs=0) for pr,rep in jobs]
    outputs=llm.generate([{'prompt_token_ids':pr['prompt_token_ids']} for pr,rep in jobs],sampling,
                        lora_request=LoRARequest('diagnostic',123,p['adapter']),use_tqdm=True)
    with target.open('x',encoding='utf-8') as f:
        for (pr,rep),output,params in zip(jobs,outputs,sampling,strict=True):
            assert list(output.prompt_token_ids)==pr['prompt_token_ids']
            for sample,seq in enumerate(output.outputs):
                reward=compute_score(pr['data_source'],seq.text,pr['gold'])
                tokens=list(seq.token_ids)
                lps=[float(x[t].logprob) for t,x in zip(tokens,seq.logprobs,strict=True)]
                record={'policy':key,'prompt_index':pr['prompt_index'],'prompt_id':pr['prompt_id'],
                    'repeat':rep,'sample':sample,'seed':params.seed,'adapter_id':123,
                    'prompt_token_ids':pr['prompt_token_ids'],'response_token_ids':tokens,
                    'rollout_logprobs':lps,'text':seq.text,'finish_reason':seq.finish_reason,
                    'reward':reward,'source_adapter_sha256':p['adapter_sha256']}
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    write(out/f'generation_done.{shard}.json',{'status':'complete','at':now(),'seconds':time.monotonic()-start,
        'records':len(jobs)*8,'rollouts_sha256':sha(target),'smoke':smoke})

def records_for(out):
    rows=[]
    for f in sorted(out.glob('rollouts.*.jsonl')):
        rows.extend(read_jsonl(f))
    return sorted(rows,key=lambda r:(r['repeat'],r['prompt_index'],r['sample']))

def make_weights(plan,key):
    import torch
    from verl.utils.seqlen_balancing import calculate_workload,get_seqlen_balanced_partitions
    out=OPS/key;rows=records_for(out)
    assert len(rows)==1024
    assert len({(r['repeat'],r['prompt_index'],r['sample']) for r in rows})==1024
    weights={};packing=[]
    for rep in range(2):
        rr=[r for r in rows if r['repeat']==rep]
        plen=[len(r['prompt_token_ids']) for r in rr]
        rlen=[len(r['response_token_ids']) for r in rr]
        seq=torch.tensor([p+r for p,r in zip(plen,rlen)])
        work=calculate_workload(seq).tolist()
        ranks=get_seqlen_balanced_partitions(work,4,True)
        for d,part in enumerate(ranks):
            part.sort(key=lambda i:(work[i],i));ranks[d]=part[::2]+part[1::2][::-1]
        for update in range(4):
            chunks=[part[update*32:(update+1)*32] for part in ranks]
            n_micro=max(math.ceil(sum(int(seq[i]) for i in c)/12288) for c in chunks)
            for d,c in enumerate(chunks):
                parts=get_seqlen_balanced_partitions([work[i] for i in c],n_micro,False)
                for m,part in enumerate(parts):
                    members=[c[i] for i in part];T=sum(rlen[i] for i in members)
                    coefficient=len(members)/(4*32*T)
                    packing.append({'repeat':rep,'update':update,'dp_rank':d,'microbatch':m,
                        'members':members,'response_count':len(members),'response_tokens':T,
                        'token_coefficient':coefficient})
                    for i in members:
                        r=rr[i]; weights[f'{rep}:{r["prompt_index"]}:{r["sample"]}']={
                            'formal_update':update,'formal_token_coefficient':coefficient}
        for pidx in range(64):
            group=[r for r in rr if r['prompt_index']==pidx]
            rew=torch.tensor([r['reward']['score'] for r in group],dtype=torch.float32)
            adv=(rew-rew.mean())/(rew.std()+1e-6)
            population=(rew-rew.mean())/(rew.std(correction=0)+1e-30)
            total=sum(len(r['response_token_ids']) for r in group)
            valid=2<=int(rew.sum())<=6
            for sample,r in enumerate(group):
                k=f'{rep}:{pidx}:{sample}'
                weights[k].update(advantage=float(adv[sample]),group_tokens=total,
                    real_valid=valid,real_coefficient=float(-population[sample]/(8*len(r['response_token_ids']))) if valid else 0.)
    # Each formal update coefficient integrates to one over valid response tokens.
    for rep in range(2):
        for u in range(4):
            total=sum(x['response_tokens']*x['token_coefficient'] for x in packing if x['repeat']==rep and x['update']==u)
            assert abs(total-1)<1e-12
    write(out/'weights.json',{'weights':weights,'packing':packing,
        'record_hashes':{str(p):sha(p) for p in out.glob('rollouts.*.jsonl')},
        'formula':'c_dj=n_dj/(D*N_d*T_dj), D=4, N_d=32; same frozen policy for all4 logical updates'})
    return weights


def load_hf(plan,p):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    from verl.utils.lora_resume_verification import canonical_tensors,tensor_fingerprint
    base=AutoModelForCausalLM.from_pretrained(plan['base_model'],torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',local_files_only=True).cuda()
    model=PeftModel.from_pretrained(base,p['adapter'],is_trainable=False)
    source=canonical_tensors(load_file(str(Path(p['adapter'])/'adapter_model.safetensors')))
    actual=canonical_tensors({k:v for k,v in model.named_parameters() if 'lora_A' in k or 'lora_B' in k})
    assert set(actual)==set(source) and len(actual)==392
    for k,v in actual.items():
        assert torch.equal(v,source[k].to(v.dtype)), k
    for x in model.parameters(): x.requires_grad_(False)
    return model,{'status':'verified','tensor_count':392,'adapter_sha256':p['adapter_sha256'],
                  'loaded_fingerprint':tensor_fingerprint(actual)}

def install_probes(model,modules):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Probe(nn.Module):
        def __init__(self,old,name):
            super().__init__();self.original=old
            din=old.in_features;dout=old.out_features
            gen=torch.Generator(device='cpu').manual_seed(seed_for('projection',name))
            b=torch.randn(dout,64,generator=gen,dtype=torch.float32)/math.sqrt(32)
            self.register_buffer('probe_B',b.cuda())
            self.probe_A=nn.Parameter(torch.zeros(64,din,device='cuda',dtype=torch.float32))
        def forward(self,x,*args,**kwargs):
            y=self.original(x,*args,**kwargs)
            return y+F.linear(F.linear(x.float(),self.probe_A),self.probe_B).to(y.dtype)
    probes={}
    for name in modules:
        full='base_model.model.'+name
        parent_name,leaf=full.rsplit('.',1);parent=model.get_submodule(parent_name)
        obj=Probe(getattr(parent,leaf),name);setattr(parent,leaf,obj);probes[name]=obj
    return probes

def nll_and_logps(model,ids,prompt_len):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    core=model.base_model.model
    hidden=core.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0,prompt_len-1:-1]
    targets=ids[0,prompt_len:]
    def chunk(h,t):
        return -F.cross_entropy(core.lm_head(h).float(),t,reduction='none')
    parts=[]
    for start in range(0,len(targets),256):
        h=hidden[start:start+256];t=targets[start:start+256]
        parts.append(checkpoint(chunk,h,t,use_reentrant=False) if torch.is_grad_enabled() else chunk(h,t))
    lp=torch.cat(parts)
    return -lp.mean(),lp.detach()

def backward(plan,key,shard,smoke=False):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    p=policy(plan,key);out=OPS/('smoke' if smoke else key)
    rows=read_jsonl(out/f'rollouts.{shard}.jsonl')
    rows.sort(key=lambda r:(r['prompt_index'],r['repeat'],r['sample']))
    weights=read(out/'weights.json')['weights'] if not smoke else None
    model,proof=load_hf(plan,p)
    first=rows[0];short=torch.tensor([first['prompt_token_ids'][-32:]+first['response_token_ids'][:8]],device='cuda')
    model.eval()
    with torch.no_grad(): before=model(short,use_cache=False).logits
    probes=install_probes(model,plan['modules'])
    with torch.no_grad(): after=model(short,use_cache=False).logits
    assert torch.equal(before,after),'Zero probe changed logits'
    proof.update(zero_probe_logits_exact=True,modules=plan['modules'])
    del before,after
    for mod in model.modules():
        if isinstance(mod,torch.nn.Dropout): assert mod.p==0
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    write(out/f'hf_verification.{shard}.json',proof)
    n=1 if smoke else 16
    equal={name:torch.zeros(2,n,64,probe.probe_A.shape[1]) for name,probe in probes.items()}
    formal={name:torch.zeros(2,4,n,64,probe.probe_A.shape[1]) for name,probe in probes.items()}
    real={name:torch.zeros(2,n,64,probe.probe_A.shape[1]) for name,probe in probes.items()}
    logps={}; stats=[];start=time.monotonic()
    local_indices=sorted({r['prompt_index'] for r in rows})
    for num,r in enumerate(rows):
        rep,idx,sample=r['repeat'],r['prompt_index'],r['sample'];local=local_indices.index(idx)
        group=[x for x in rows if x['repeat']==rep and x['prompt_index']==idx]
        if smoke:
            rewards=torch.tensor([x['reward']['score'] for x in group])
            adv=float(((rewards-rewards.mean())/(rewards.std()+1e-6))[sample])
            # Exercise a nonzero backward even if the short smoke answers all fail.
            w={'advantage':adv if adv else (1. if sample%2 else -1.),
               'group_tokens':sum(len(x['response_token_ids']) for x in group),
               'formal_update':0,'formal_token_coefficient':1/8192,'real_coefficient':0.}
        else: w=weights[f'{rep}:{idx}:{sample}']
        ids=torch.tensor([r['prompt_token_ids']+r['response_token_ids']],device='cuda')
        for probe in probes.values(): probe.probe_A.grad=None
        active=w['advantage']!=0 or (idx<32 and w['real_coefficient']!=0)
        with torch.set_grad_enabled(active):
            loss,lps=nll_and_logps(model,ids,len(r['prompt_token_ids']))
            if active: loss.backward()
        L=len(r['response_token_ids'])
        # One response-mean NLL backward supports all scalar loss/reduction weights.
        if active:
            for name,probe in probes.items():
                g=probe.probe_A.grad.detach().cpu()
                assert torch.isfinite(g).all(),(name,idx,rep,sample)
                equal[name][rep,local].add_(g,alpha=w['advantage']*L/w['group_tokens'])
                formal[name][rep,w['formal_update'],local].add_(g,alpha=w['advantage']*L*w['formal_token_coefficient'])
                if idx<32:
                    # REAL gradient is +contrast*logp, hence -coefficient*NLL.
                    real[name][rep,local].add_(g,alpha=L*w['real_coefficient'])
        logps[f'{rep}_{idx}_{sample}']=lps.float().cpu().numpy()
        stats.append({'repeat':rep,'prompt_index':idx,'sample':sample,'tokens':L,'mean_nll':float(loss.detach()),
                      'advantage':w['advantage'],'backward':active})
        write(out/f'backward_status.{shard}.json',{'status':'running','at':now(),'completed':num+1,
              'total':len(rows),'seconds':time.monotonic()-start,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30})
    if not smoke and shard>=2:
        real={name:x[:,:0].clone() for name,x in real.items()}
    torch.save({'equal':equal,'formal':formal,'real':real,'prompt_indices':local_indices},out/f'sketches.{shard}.pt')
    np.savez(out/f'hf_logprobs.{shard}.npz',**logps)
    write(out/f'backward_done.{shard}.json',{'status':'complete','at':now(),'seconds':time.monotonic()-start,
         'stats':stats,'sketches_sha256':sha(out/f'sketches.{shard}.pt'),'smoke':smoke})


def run_stage(plan,action,key):
    children=[]; handles=[]
    try:
        for shard in range(4):
            marker=OPS/key/f'{"generation" if action=="generate" else "backward"}_done.{shard}.json'
            if marker.exists():
                evidence=read(marker)
                artifact=OPS/key/(f'rollouts.{shard}.jsonl' if action=='generate' else f'sketches.{shard}.pt')
                expected=evidence['rollouts_sha256' if action=='generate' else 'sketches_sha256']
                assert evidence['status']=='complete' and sha(artifact)==expected
                continue
            log=OPS/key/f'{action}.{shard}.log';log.parent.mkdir(exist_ok=True)
            h=log.open('a',encoding='utf-8');handles.append(h)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH=str(ROOT),
                TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
            proc=subprocess.Popen([plan['python'],'-u',str(Path(__file__).resolve()),action,'--policy',key,'--shard',str(shard)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(proc)
        while any(p.poll() is None for p in children):
            bad=[p for p in children if p.poll() not in (None,0)]
            if bad: raise RuntimeError(f'{action}/{key} shard failed: {[p.returncode for p in children]}')
            write(OPS/'status.json',{'status':'running','at':now(),'policy':key,'stage':action,
                  'children':[{'pid':p.pid,'returncode':p.poll()} for p in children]})
            time.sleep(15)
        assert all(p.returncode==0 for p in children)
    except BaseException:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        raise
    finally:
        for h in handles:h.close()

def run(plan):
    import fcntl
    import shutil
    with ExitStack() as stack:
        for p in [str(OPS/'controller.lock')]+plan['resource_locks']:
            h=stack.enter_context(open(p,'a'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OPS/'math_checks.json')['status']=='passed'
        assert read(OPS/'analysis_checks.json')['status']=='passed'
        assert read(OPS/'smoke/backward_done.0.json')['status']=='complete'
        assert read(OPS/'smoke/hf_verification.0.json')['zero_probe_logits_exact']
        assert read(OPS/'smoke/engine_verification.0.json')['status']=='verified'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPUs are occupied'
        write(OPS/'launch.json',{'at':now(),'pid':os.getpid(),'plan_sha256':sha(OPS/'plan.json')})
        try:
            for p in plan['policies']:
                key=p['key'];out=OPS/key;out.mkdir(exist_ok=True)
                load_plan()
                if (out/'analysis.json').exists():
                    assert read(out/'analysis.json')['status']=='complete'
                    continue
                from safetensors import safe_open
                with safe_open(plan['subspaces']['gradtop_init'],framework='pt') as f:
                    input_dims=sum(f.get_slice(name).get_shape()[1] for name in plan['modules'])
                # Peak task-created raw arrays: equal1 + formal4 + REAL0.5;
                # before reduction cleanup, another equal1 is retained.
                equal_bytes=2*64*64*input_dims*4
                missing_rollouts=not all((out/f'generation_done.{s}.json').exists() for s in range(4))
                required=math.ceil(6.5*equal_bytes)+2*2**30+(int(.75*2**30) if missing_rollouts else 0)
                available=shutil.disk_usage(ROOT).free
                write(out/'disk_preflight.json',{'at':now(),'required_free_bytes':required,
                    'available_free_bytes':available,'headroom_gib':2,'new_rollouts_needed':missing_rollouts})
                assert available>required, f'Insufficient disk for {key}: {available} <= {required}'
                if not all((out/f'generation_done.{s}.json').exists() for s in range(4)):
                    run_stage(plan,'generate',key)
                make_weights(plan,key)
                if not (out/'analysis.json').exists():
                    run_stage(plan,'backward',key)
                    subprocess.run([plan['python'],str(ROOT/'scripts/analysis/summarize_gradtop_real_diagnostics.py'),
                                    '--policy',key],cwd=ROOT,check=True)
            subprocess.run([plan['python'],str(ROOT/'scripts/analysis/summarize_gradtop_real_diagnostics.py'),
                            '--all'],cwd=ROOT,check=True)
            write(OPS/'status.json',{'status':'complete','at':now(),'policies':[p['key'] for p in plan['policies']],
                                    'results':str(OPS/'results.md')})
        except BaseException as e:
            write(OPS/'status.json',{'status':'error','at':now(),'error':repr(e)});raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','generate','weights','backward','run'])
    parser.add_argument('--policy',default='gradtop100');parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.action=='register':register();return
    p=load_plan()
    if args.action=='generate':generate(p,args.policy,args.shard,args.smoke)
    elif args.action=='weights':make_weights(p,args.policy)
    elif args.action=='backward':backward(p,args.policy,args.shard,args.smoke)
    else:run(p)

if __name__=='__main__':main()
