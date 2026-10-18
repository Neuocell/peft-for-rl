#!/usr/bin/env python3
"""Reduce paired sketches to reproducible Gram evidence and diagnostic tables."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from run_gradtop_real_diagnostics import OPS,read,read_jsonl,write,sha,now,load_plan,policy
from verl.utils.lora_resume_verification import canonical_tensors


def bases(plan,p,name):
    result={}
    for key,path in plan['subspaces'].items():
        a=load_file(path)[name].float()
        result[key]=torch.linalg.qr(a.T,mode='reduced')[0]
    adapter=canonical_tensors(load_file(str(Path(p['adapter'])/'adapter_model.safetensors')))
    a=adapter[name+'.lora_A.weight'].float()
    result['current_A']=torch.linalg.qr(a.T,mode='reduced')[0]
    for q in result.values():
        assert torch.allclose(q.T@q,torch.eye(32),atol=2e-5)
    return result


def grams(x,qs):
    # x has repeat,prompt,width,input. Prompt order is shared across repeats.
    out={}
    for half,sl in [('A',slice(0,32)),('B',slice(32,64))]:
        y=x[:,:,sl,:].reshape(128,32,x.shape[-1])
        flat=y.reshape(128,-1)
        out[half+'/full']=(flat@flat.T).double().numpy()
        for key,q in qs.items():
            z=(y@q).reshape(128,-1)
            out[half+'/'+key]=(z@z.T).double().numpy()
    return out


def summary(g,group_weights=None):
    a=g[:64,:64];b=g[64:,64:];ab=g[:64,64:]
    u=float((np.trace(a)+np.trace(b))/128)
    s=float(np.trace(ab)/64)
    m=float(ab.mean());noise=u-s;heterogeneity=s-m
    shared=[];cross=[]
    for start in range(0,64,16):
        ix=np.arange(start,start+16)
        if group_weights is None:
            wa=wb=np.full(16,1/16)
        else:wa=group_weights[0,ix];wb=group_weights[1,ix]
        shared.extend([float(wa@a[np.ix_(ix,ix)]@wa),float(wb@b[np.ix_(ix,ix)]@wb)])
        cross.append(float(wa@ab[np.ix_(ix,ix)]@wb))
    denom=np.sqrt(np.diag(a)*np.diag(b))
    nonzero=denom>1e-30
    return {'individual_energy':u,'repeat_cross_energy':s,'noise_energy':noise,
        'shared_mean_cross_energy':m,'heterogeneity_cross_energy':heterogeneity,
        'shared16_energy':float(np.mean(shared)), 'shared16_repeat_cross_energy':float(np.mean(cross)),
        'repeat_cross_fraction':s/u if u>1e-30 else None,
        'repeat_cosine_mean_nonzero':float(np.mean(np.diag(ab)[nonzero]/denom[nonzero])) if nonzero.any() else None,
        'nonzero_repeat_pairs':int(nonzero.sum()),
        'zero_individual_groups':int((np.diag(g)<=1e-30).sum())}


def bootstrap_coverage(g,projected,seed=42):
    rng=np.random.default_rng(seed);inds=rng.integers(0,64,size=(2000,64))
    d=np.diag(g);v=np.diag(projected)
    denom=(d[inds]+d[inds+64]).sum(1);num=(v[inds]+v[inds+64]).sum(1)
    values=num[denom>1e-30]/denom[denom>1e-30]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else None


def bootstrap_signal(g):
    rng=np.random.default_rng(42);ix=rng.integers(0,64,size=(2000,64))
    total=(np.diag(g[:64,:64])+np.diag(g[64:,64:]))/2
    signal=np.diag(g[:64,64:])
    denominator=total[ix].mean(1);s=signal[ix].mean(1)
    valid=denominator>1e-30
    return {'cross_energy_ci95':np.quantile(s,[.025,.975]).tolist(),
            'repeat_cross_fraction_ci95':np.quantile(s[valid]/denominator[valid],[.025,.975]).tolist() if valid.any() else None}


def estimator_audit(x):
    """Same stream and objective, one-shot vs capacity64 incremental SVD.

    Four16-prompt observations, output width32; fit repeat0/projectionA,
    evaluate repeat1/projectionB. This measures the solver, not deployed width8 speed.
    """
    import time
    blocks=[x[0,i:i+16,:32].mean(0).T for i in range(0,64,16)]
    stream=torch.cat(blocks,dim=1)
    audit=x[1,:,32:]
    audit_batches=audit.reshape(4,16,32,x.shape[-1]).mean(1)
    q_ref=None;out=[]
    for method,order in [('one_shot',list(range(4))),('incremental64',list(range(4))),('reverse_incremental64',list(reversed(range(4))))]:
        start=time.monotonic()
        if method=='one_shot':
            q,r=torch.linalg.qr(stream,mode='reduced');u,s,_=torch.linalg.svd(r,full_matrices=False)
            basis=(q@u)[:,:32]
        else:
            f=stream[:,:0]
            for i in order:
                q,r=torch.linalg.qr(torch.cat([f,blocks[i]],dim=1),mode='reduced')
                u,s,_=torch.linalg.svd(r,full_matrices=False)
                f=(q@u[:,:64])*s[:64]
            basis=torch.linalg.qr(f,mode='reduced')[0][:,:32]
        if q_ref is None:q_ref=basis
        den=float(audit.square().sum());bden=float(audit_batches.square().sum())
        out.append({'method':method,'seconds':time.monotonic()-start,'fit_projection':'A','audit_projection':'B',
            'fit_repeat':0,'audit_repeat':1,'width':32,'capacity':64,'rank':32,
            'same_stream_overlap_to_one_shot':float((q_ref.T@basis).square().sum()/32),
            'audit_individual_retention':float((audit@basis).square().sum())/den if den>1e-30 else None,
            'audit_shared16_retention':float((audit_batches@basis).square().sum())/bden if bden>1e-30 else None,
            'signal_rank':int((s>s[0]*1e-6).sum()) if float(s[0])>0 else 0})
    return out


def analyze(key):
    torch.set_num_threads(16)
    plan=load_plan();p=policy(plan,key);out=OPS/key
    assert not (out/'analysis.json').exists()
    shards=[torch.load(out/f'sketches.{s}.pt',map_location='cpu',weights_only=True) for s in range(4)]
    assert [i for d in shards for i in d['prompt_indices']]==list(range(64))
    eqdir=out/'equal_sketches';eqdir.mkdir(exist_ok=True)
    w=read(out/'weights.json')['weights']
    lens=np.array([[w[f'{rep}:{i}:0']['group_tokens'] for i in range(64)] for rep in range(2)])
    norm=np.empty_like(lens,dtype=float)
    for rep in range(2):
        for start in range(0,64,16):
            norm[rep,start:start+16]=lens[rep,start:start+16]/lens[rep,start:start+16].sum()
    all_grams={};metrics=[];formal_totals={};retained={};estimators=[];real_metrics=[]
    for name in plan['modules']:
        slug=name.replace('.','_');qs=bases(plan,p,name)
        equal=torch.cat([d['equal'][name] for d in shards],dim=1)
        np.save(eqdir/f'{slug}.npy',equal.numpy())
        retained[str(eqdir/f'{slug}.npy')]=sha(eqdir/f'{slug}.npy')
        estimators.extend([dict(module=name,**r) for r in estimator_audit(equal)])
        gg=grams(equal,qs)
        global_x=equal*torch.tensor(norm*16,dtype=torch.float32)[:,:,None,None]
        global_grams=grams(global_x,qs)
        for branch,gs in [('equal',gg),('global_token',global_grams)]:
            for k,g in gs.items():all_grams[f'{slug}/{branch}/{k}']=g
            for half in ('A','B'):
                ref=summary(gs[half+'/full'])
                for space in ['full',*qs]:
                    z=summary(gs[half+'/'+space])
                    if branch=='equal':z.update(bootstrap_signal(gs[half+'/'+space]))
                    for metric in ('individual_energy','shared16_energy'):
                        z[metric+'_retention']=z[metric]/ref[metric] if ref[metric]>1e-30 else None
                    z.update(module=name,branch=branch,projection=half,space=space)
                    if space!='full':z['individual_retention_ci95']=bootstrap_coverage(gs[half+'/full'],gs[half+'/'+space])
                    metrics.append(z)
        for update in range(4):
            x=torch.cat([d['formal'][name][:,update] for d in shards],dim=1)
            gs=grams(x,qs)
            for k,g in gs.items():
                all_grams[f'{slug}/formal{update}/{k}']=g
                formal_totals.setdefault((name,k),[]).append(g)
        # REAL original objective sensitivity, only first32 selected prompts; rejected groups remain flagged zeros.
        rx=torch.cat([d['real'][name] for d in shards],dim=1)[:,:32]
        for half,sl in [('A',slice(0,32)),('B',slice(32,64))]:
            y=rx[:,:,sl,:].reshape(64,32,rx.shape[-1])
            valid=[i for i in range(32) if w[f'0:{i}:0']['real_valid'] and w[f'1:{i}:0']['real_valid']]
            ii=valid+[i+32 for i in valid]
            full=y.reshape(64,-1)
            denom=float(full[ii].square().sum()) if ii else 0.
            for space,q in [('full',None),*qs.items()]:
                z=y if q is None else y@q;flat=z.reshape(64,-1)
                all_grams[f'{slug}/real_first32/{half}/{space}']=(flat@flat.T).double().numpy()
                if ii:
                    num=float(flat[ii].square().sum())
                    cross=float((flat[valid]*flat[[i+32 for i in valid]]).sum())/len(valid)
                else:num=cross=0.
                real_metrics.append({'module':name,'projection':half,'space':space,'valid_both_prompts':len(valid),
                    'individual_retention':num/denom if denom>1e-30 else None,'repeat_cross_energy':cross})
    # Formal update members differ by response length and may split prompt groups.
    # Sum the actual partial contributions for each update; no fictitious complete16-prompt groups.
    for name in plan['modules']:
        for half in ('A','B'):
            ref=formal_totals[(name,half+'/full')]
            for space in ('full','gradtop_init','real_init','current_A'):
                gs=formal_totals[(name,half+'/'+space)]
                individual=float(sum(np.trace(g) for g in gs)/128)
                batch=float(np.mean([g[:64,:64].sum() for g in gs]+[g[64:,64:].sum() for g in gs]))
                ref_ind=float(sum(np.trace(g) for g in ref)/128)
                ref_batch=float(np.mean([g[:64,:64].sum() for g in ref]+[g[64:,64:].sum() for g in ref]))
                cross=float(np.mean([g[:64,64:].sum() for g in gs]))
                metrics.append({'module':name,'branch':'formal','projection':half,'space':space,
                    'individual_energy':individual,'shared16_energy':batch,
                    'individual_energy_retention':individual/ref_ind if ref_ind>1e-30 else None,
                    'shared16_energy_retention':batch/ref_batch if ref_batch>1e-30 else None,
                    'shared16_repeat_cross_energy':cross,
                    'formal_repeat_caveat':'update memberships and token coefficients differ across repeat rollouts'})
    np.savez(out/'gram_evidence.npz',**all_grams)
    rows=[]
    for f in out.glob('rollouts.*.jsonl'):rows.extend(read_jsonl(f))
    behavior={'reward':float(np.mean([r['reward']['score'] for r in rows])),
              'parse':float(np.mean([r['reward']['parse_success'] for r in rows])),
              'length':float(np.mean([len(r['response_token_ids']) for r in rows])),
              'hit_max':float(np.mean([r['finish_reason']=='length' for r in rows])),
              'zero_advantage_groups':sum(w[f'{rep}:{i}:0']['advantage']==0 for rep in range(2) for i in range(64)),
              'real_valid_both_first32':sum(w[f'0:{i}:0']['real_valid'] and w[f'1:{i}:0']['real_valid'] for i in range(32))}
    done={'status':'complete','at':now(),'policy':p,'behavior':behavior,'metrics':metrics,
          'gram_sha256':sha(out/'gram_evidence.npz'),'retained_sketches_sha256':retained,
          'raw_shards_sha256':{str(out/f'sketches.{s}.pt'):sha(out/f'sketches.{s}.pt') for s in range(4)},
          'raw_shards_reduced_to_gram_evidence':True,'estimator_audit':estimators,'real_sensitivity':real_metrics}
    write(out/'analysis.json',done)
    # Only task-created, reproducibly reduced temporary formal sketches are removed.
    for s in range(4):(out/f'sketches.{s}.pt').unlink()
    report(False)


def trajectory(plan):
    """Matched-prompt bootstrap and cross-policy mean-gradient rotation evidence."""
    pairs=[('gradtop100','gradtop150'),('real100','real150'),('gradtop150','gradtop270')]
    results=[];rng=np.random.default_rng(42);ix=rng.integers(0,64,size=(2000,64))
    for before,after in pairs:
        ga=np.load(OPS/before/'gram_evidence.npz');gb=np.load(OPS/after/'gram_evidence.npz')
        rows=[];rotation=[]
        for branch in ('equal','global_token'):
            for space in ('gradtop_init','real_init'):
                delta_samples=[];point=[]
                for name in plan['modules']:
                    slug=name.replace('.','_')
                    for half in ('A','B'):
                        values=[];boots=[]
                        for gram in (ga,gb):
                            den=np.diag(gram[f'{slug}/{branch}/{half}/full'])
                            num=np.diag(gram[f'{slug}/{branch}/{half}/{space}'])
                            d=den[:64]+den[64:];n=num[:64]+num[64:]
                            values.append(n.sum()/d.sum() if d.sum()>1e-30 else 0.)
                            bd=d[ix].sum(1);bn=n[ix].sum(1)
                            boots.append(np.divide(bn,bd,out=np.zeros_like(bn),where=bd>1e-30))
                        point.append(values[1]-values[0]);delta_samples.append(boots[1]-boots[0])
                meanboot=np.mean(delta_samples,axis=0)
                rows.append({'branch':branch,'space':space,'individual_retention_delta':float(np.mean(point)),
                    'paired_prompt_ci95':np.quantile(meanboot,[.025,.975]).tolist(),'bootstrap_samples':2000})
        for name in plan['modules']:
            slug=name.replace('.','_')
            a=np.load(OPS/before/'equal_sketches'/f'{slug}.npy',mmap_mode='r')
            b=np.load(OPS/after/'equal_sketches'/f'{slug}.npy',mmap_mode='r')
            for half,sl in [('A',slice(0,32)),('B',slice(32,64))]:
                am=np.asarray(a[:,:,sl,:]).mean(axis=1,dtype=np.float64)
                bm=np.asarray(b[:,:,sl,:]).mean(axis=1,dtype=np.float64)
                def cos(x,y):
                    den=float(np.linalg.norm(x)*np.linalg.norm(y))
                    return float((x*y).sum()/den) if den>1e-30 else None
                rotation.append({'module':name,'projection':half,
                    'within_before_repeat_mean_cosine':cos(am[0],am[1]),
                    'within_after_repeat_mean_cosine':cos(bm[0],bm[1]),
                    'cross_policy_pooled_mean_cosine':cos(am.mean(0),bm.mean(0))})
        results.append({'before':before,'after':after,'coverage_changes':rows,'mean_gradient_rotation':rotation})
    write(OPS/'trajectory.json',{'at':now(),'comparisons':results,
        'interpretation':'paired fixed prompts, two independent rollout groups; conditional on single training trajectory and fixed diagnostic projections'})


def report(final):
    plan=load_plan();results=[]
    for p in plan['policies']:
        f=OPS/p['key']/'analysis.json'
        if f.exists():results.append(read(f))
    if final:
        assert len(results)==len(plan['policies'])
        trajectory(plan)
    table=['# GradTop / REAL 固定策略梯度诊断','',f'更新时间：{now()}。完成{len(results)}/5个策略状态。',
           '', '64个固定prompts、两组独立K8；采样temperature1/top-p1、8192输出上限。',
           '模块：第0/13/27层q_proj与down_proj，共6个。两组独立输出sketch各宽32；正式训练optimizer更新数为0。',
           '', '| Policy | Reward | Parse | 平均长度 | Hit-max | 零advantage组/128 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        b=r['behavior'];table.append(f"| {r['policy']['key']} | {100*b['reward']:.2f}% | {100*b['parse']:.2f}% | {b['length']:.1f} | {100*b['hit_max']:.2f}% | {b['zero_advantage_groups']} |")
    table.extend(['','下表对6个模块、两个独立sketch取等权均值；百分比是各自梯度能量保留率，不是准确率。完整分层值及逐题证据见各policy的analysis.json/gram_evidence.npz。',
                  '', '| Policy | 归约 | 候选子空间 | 逐题/partial-prompt能量保留 | 共享更新能量保留 |',
                  '| --- | --- | --- | ---: | ---: |'])
    for r in results:
        for branch in ('equal','global_token','formal'):
            for space in ('gradtop_init','real_init','current_A'):
                ms=[m for m in r['metrics'] if m['branch']==branch and m['space']==space]
                i=np.mean([m['individual_energy_retention'] for m in ms if m['individual_energy_retention'] is not None])
                b=np.mean([m['shared16_energy_retention'] for m in ms if m['shared16_energy_retention'] is not None])
                table.append(f"| {r['policy']['key']} | {branch} | {space} | {100*i:.3f}% | {100*b:.3f}% |")
    table.extend(['','| Policy | 全空间可重复逐题能量 / 总能量 | 双组非零梯度余弦 |',
                  '| --- | ---: | ---: |'])
    for r in results:
        ms=[m for m in r['metrics'] if m['branch']=='equal' and m['space']=='full']
        signal=[m['repeat_cross_fraction'] for m in ms if m['repeat_cross_fraction'] is not None]
        cosine=[m['repeat_cosine_mean_nonzero'] for m in ms if m['repeat_cosine_mean_nonzero'] is not None]
        table.append(f"| {r['policy']['key']} | {np.mean(signal):.4f} | {np.mean(cosine):.4f} |")
    table.extend(['','跨重复能量允许为负，不截断为正；噪声与异质性分解及区间保留在analysis.json。',
                  '增量SVD审计固定同一统计目标和同一sketch流，拟合repeat0/projectionA，评价repeat1/projectionB；诊断width32不同于正式GradTop probe width8，其耗时不能当作正式probe成本。',
                  'REAL原目标敏感性保留在前32题的Gram中；有效组按两次都满足至少2正2负限定。原token与系数保留，可重新反传。',
                  'formal分支复现长度均衡、四DP划分和动态microbatch权重，但所有四个逻辑update冻结在同一个policy，不能称为历史Adam更新的逐位复现。',
                  '当前轨迹51/101的旧恢复异常仍存在；新诊断加载采用源权重→vLLM注册及CUDA槽位、HF adapter的独立检查。',
                  'REAL270没有checkpoint，本轮缺失该点；诊断题来自训练源且排除probe前4096题，不是独立测试集。'])
    (OPS/'results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write(OPS/'results.json',{'status':'complete' if final else 'partial','at':now(),
        'completed':[r['policy']['key'] for r in results], 'analyses':{r['policy']['key']:sha(OPS/r['policy']['key']/'analysis.json') for r in results}})

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--policy');pa.add_argument('--all',action='store_true');args=pa.parse_args()
    if args.all:report(True)
    else:analyze(args.policy)
