"""MORE: single-dataset, fixed-epoch unsupervised graph anomaly detection."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from graph_io import load_graph,discover,make_split,reference_ranks,metrics,write_csv
from bounded_graph import GraphPlan,calibration
from fast_utils import estimate_gpu_bytes,normalized_operator,sample_constraints,feature_audit


class Detector(nn.Module):
    def __init__(self,d,hidden,embedding):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,hidden),nn.LeakyReLU(.2),nn.Linear(hidden,embedding))
    def forward(self,x):return self.net(x)


def tensor_input(x,ids,device):
    values=x[ids]
    if sp.issparse(values):values=values.toarray()
    return torch.as_tensor(np.asarray(values),dtype=torch.float32,device=device)


def tensor_op(op,device):
    coo=op.tocoo()
    indices=torch.as_tensor(np.vstack((coo.row,coo.col)),dtype=torch.long,device=device)
    values=torch.as_tensor(coo.data,dtype=torch.float32,device=device)
    return torch.sparse_coo_tensor(indices,values,op.shape,device=device).coalesce()


class Engine:
    """Exact GPU graph when it fits; otherwise deterministic bounded cached parts.

    The budget is a planning envelope, not a promise about CUDA workspace use.
    CUDA allocator memory is NOT capped to 10 GiB; a 24 GiB card retains headroom.
    """
    def __init__(self,x,a,args,folder):
        self.x=x;self.a=a;self.args=args;self.cache=OrderedDict();self.cache_bytes=0
        self.device=torch.device(args.device if args.device!='auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        budget=int(args.gpu_budget_gb*1024**3)
        if self.device.type=='cuda':
            free,_=torch.cuda.mem_get_info(self.device)
            budget=min(budget,int(.8*free))
        self.budget=budget
        estimate=estimate_gpu_bytes(x.shape[0],x.shape[1],a.nnz,args.hidden_dim,args.embedding_dim,args.order)
        full=args.graph_mode=='full' or (args.graph_mode=='auto' and estimate<=budget)
        if full and estimate>budget:
            raise MemoryError('Full graph planning estimate exceeds budget; use --graph-mode auto or bounded.')
        self.mode='full' if full else 'bounded'
        self.global_x=None;self.full_batch=None
        self.cache_cap=min(int(args.cache_gb*1024**3),int(.45*budget))
        if full:
            # Stream CPU feature blocks into one resident GPU tensor.
            data=torch.empty((x.shape[0],x.shape[1]),dtype=torch.float32,device=self.device)
            for start in range(0,x.shape[0],args.transfer_rows):
                data[start:start+args.transfer_rows].copy_(tensor_input(x,slice(start,start+args.transfer_rows),self.device))
            op=tensor_op(normalized_operator(a),self.device)
            ids=np.arange(x.shape[0]);pos=torch.arange(x.shape[0],device=self.device)
            self.full_batch=(ids,data,pos,op)
            self.nparts=1;self.plan=None
            self.report=dict(mode='full',exact=True,parts=1,nodes=x.shape[0],edges=a.nnz,
                             planning_estimate_gib=estimate/1024**3,budget_gib=budget/1024**3)
        else:
            params=copy.copy(args)
            # Leave space for cache, active autograd tensors and optimizer state.
            while estimate_gpu_bytes(params.halo_nodes,x.shape[1],params.halo_edges,
                                     args.hidden_dim,args.embedding_dim,args.order)>.5*budget:
                if params.halo_nodes<=32 or params.halo_edges<=64:
                    raise MemoryError('Planning budget is too small even for a minimal subgraph')
                params.halo_nodes=max(32,params.halo_nodes//2)
                params.halo_edges=max(64,params.halo_edges//2)
                params.core_nodes=min(params.core_nodes,params.halo_nodes)
            self.plan=GraphPlan(a,x.shape[1],params,folder/'graph_cache')
            self.nparts=len(self.plan.paths)
            self.report=dict(self.plan.report,mode='bounded',budget_gib=budget/1024**3,
                             effective_feature_mb=params.feature_mb)
            # Global feature cache avoids repeated sparse-to-dense conversion if cheap enough.
            xbytes=4*x.shape[0]*x.shape[1]
            if xbytes<min(self.cache_cap//2,budget//5):
                self.global_x=torch.empty((x.shape[0],x.shape[1]),dtype=torch.float32,device=self.device)
                for start in range(0,x.shape[0],args.transfer_rows):
                    self.global_x[start:start+args.transfer_rows].copy_(tensor_input(x,slice(start,start+args.transfer_rows),self.device))
                self.cache_cap-=xbytes
        if self.mode=='bounded':
            print('Bounded execution may truncate graph responses and restrict triplets to each core; '
                  'use --graph-mode full for the paper formulation.',flush=True)
        (folder/'execution_plan.json').write_text(json.dumps(self.report,indent=2),encoding='utf-8')
        print(f'Execution: {self.mode}, parts={self.nparts}, budget={budget/1024**3:.2f} GiB, device={self.device}',flush=True)

    def core(self,part):
        if self.mode=='full':return self.full_batch[0]
        # Only load core IDs when sampling constraints, not the whole sparse operator.
        with np.load(self.plan.paths[part]) as f:return f['core']

    def get(self,part):
        if self.mode=='full':return self.full_batch
        if part in self.cache:
            entry,size=self.cache.pop(part);self.cache[part]=(entry,size)
            return entry
        core,halo,pos,op=self.plan.load(part)
        estimated=4*len(halo)*self.x.shape[1]+20*op.nnz+8*len(pos)
        while self.cache and self.cache_bytes+estimated>self.cache_cap:
            _,(_,oldsize)=self.cache.popitem(last=False);self.cache_bytes-=oldsize
        data=(self.global_x[torch.as_tensor(halo,device=self.device)] if self.global_x is not None
              else tensor_input(self.x,halo,self.device))
        entry=(core,data,torch.as_tensor(pos,device=self.device),tensor_op(op,self.device))
        if estimated<=self.cache_cap:
            self.cache[part]=(entry,estimated);self.cache_bytes+=estimated
        return entry

    def clear_cache(self):
        self.cache.clear();self.cache_bytes=0


def forward_raw(model,batch,order):
    core,x,pos,op=batch
    z=model(x);p2=z;p1=torch.sparse.mm(op,z)
    energy=[]
    for k in range(order):
        if k==0:r=p1
        else:
            r=2*torch.sparse.mm(op,p1)-p2
            p2,p1=p1,r
        energy.append(torch.log1p(r[pos].square().sum(1)))
    return z[pos],torch.stack(energy,dim=1)


def online_score(raw,center,scale,clip):
    return ((raw-center).abs()/scale).clamp(max=clip).max(1).values


@torch.no_grad()
def evaluate(model,engine,train,args):
    model.eval()
    raw=np.empty((engine.a.shape[0],args.order),np.float32)
    for part in range(engine.nparts):
        batch=engine.get(part)
        _,value=forward_raw(model,batch,args.order)
        raw[batch[0]]=value.cpu().numpy()
        del batch,value
    if not np.isfinite(raw).all():raise FloatingPointError('Non-finite teacher/inference energy')
    ev=calibration(raw,train,args.clip,args.mad_floor)
    ev['rank']=reference_ranks(ev['score'],train)
    return ev


def build_pools(engine,train,teacher,kind,args,seed,refresh):
    mask=np.zeros(engine.a.shape[0],bool);mask[train]=True
    nonisolated=np.diff(engine.a.indptr)>0
    pools=[];accepted=0;attempted=0
    for part in range(engine.nparts):
        core=engine.core(part);ids=core[mask[core]&nonisolated[core]]
        triples,weights,trials=sample_constraints(ids,engine.a,teacher['rank'],kind,
            args.pool_size,args.rank_gap,args.same_gap,args.normal_quantile,args.sampling_rounds,
            np.random.default_rng(seed+refresh*100003+part*1009))
        local=np.searchsorted(core,triples)
        pools.append((local,weights));accepted+=len(local);attempted+=trials
    return pools,accepted,attempted


def compute_losses(z,score,idx,w,train_mask,variant,args):
    zero=score.sum()*0;rank=zero;hom=zero
    gap=None;dpos=None
    if len(idx):
        if idx.shape[1]==2:
            gap=score[idx[:,0]]-score[idx[:,1]]
        else:
            aa,pp,nn=idx.T
            gap=score[nn]-torch.maximum(score[aa],score[pp])
            dpos=(score[aa]-score[pp]).abs()
            if variant=='homo':
                if args.hom_loss=='triplet':
                    dn=torch.minimum((score[nn]-score[aa]).abs(),(score[nn]-score[pp]).abs())
                    item=F.softplus((args.hom_margin+dpos-dn)/args.temperature)*args.temperature
                else:item=F.smooth_l1_loss(score[aa],score[pp],reduction='none')
                hom=(w*item).sum()/w.sum()
        rank=(w*F.softplus((args.rank_margin-gap)/args.temperature)*args.temperature).sum()/w.sum()
    zz=z[train_mask]
    var=F.relu(args.variance_target-torch.sqrt(zz.var(0,unbiased=False)+1e-4)).mean() if len(zz)>1 else zero
    return rank,hom,var,gap,dpos


def health(ev):
    return dict(mad=ev['mad'].tolist(),center=ev['center'].tolist(),
        unique_scores=int(len(np.unique(ev['score']))),
        max_score=float(ev['score'].max()),mean_score=float(ev['score'].mean()))


def run_variant(engine,y,train,val,test,initial,initial_ev,args,seed,variant,folder):
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    if engine.device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=Detector(engine.x.shape[1],args.hidden_dim,args.embedding_dim).to(engine.device)
    model.load_state_dict(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    kind='pairs' if variant=='energy_only' else 'triples'
    train_mask=np.zeros(len(y),bool);train_mask[train]=True
    ev=initial_ev;reference=initial_ev;last_eval=0
    hist=[];rows=[dict(epoch=0,**health(ev))]
    start=time.perf_counter();teacher_seconds=0.;sampling_seconds=0.;training_seconds=0.
    initial_vector=torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
    if engine.device.type=='cuda':torch.cuda.reset_peak_memory_stats(engine.device)
    for epoch in range(1,args.epochs+1):
        refresh=(epoch-1)%args.refresh_every==0
        if refresh:
            if epoch>1 and last_eval!=epoch-1:
                tick=time.perf_counter();ev=evaluate(model,engine,train,args)
                teacher_seconds+=time.perf_counter()-tick;last_eval=epoch-1
            if args.teacher_mode=='online':reference=ev
            tick=time.perf_counter()
            if variant!='var_only':
                pools,accepted,attempted=build_pools(engine,train,reference,kind,args,seed,epoch)
            else:
                pools=[(np.empty((0,3),np.int64),np.empty(0,np.float32)) for _ in range(engine.nparts)]
                accepted=attempted=0
            sampling_seconds+=time.perf_counter()-tick
            center=torch.as_tensor(ev['center'],device=engine.device)
            scale=torch.as_tensor(ev['scale'],device=engine.device)
        model.train();tick=time.perf_counter()
        losses_total=np.zeros(3,np.float64);updates=0;used=0
        coverage=np.zeros(len(y),bool);gap_mean=[];pos_mean=[]
        rng=np.random.default_rng(seed+epoch*100003)
        for _ in range(args.steps_per_epoch):
            for part in rng.permutation(engine.nparts):
                batch=engine.get(int(part));core=batch[0]
                if not train_mask[core].any():
                    del batch
                    continue
                local,w=pools[part]
                if len(local)>args.constraints_per_step:
                    selection=rng.choice(len(local),args.constraints_per_step,replace=False)
                    local,w=local[selection],w[selection]
                coverage[core[local.ravel()]]=True;used+=len(local)
                idx=torch.as_tensor(local,dtype=torch.long,device=engine.device)
                weight=torch.as_tensor(w,dtype=torch.float32,device=engine.device)
                local_train=torch.as_tensor(train_mask[core],device=engine.device)
                optimizer.zero_grad(set_to_none=True)
                z,raw=forward_raw(model,batch,args.order)
                score=online_score(raw,center,scale,args.clip)
                lr,lh,lv,gap,dp=compute_losses(z,score,idx,weight,local_train,variant,args)
                loss=args.lambda_rank*lr+args.lambda_hom*lh+args.lambda_var*lv
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                optimizer.step()
                losses_total+=np.array([float(lr.detach()),float(lh.detach()),float(lv.detach())])
                if gap is not None:gap_mean.append(float(gap.detach().mean()))
                if dp is not None:pos_mean.append(float(dp.detach().mean()))
                updates+=1
                del batch,z,raw,score,loss,lr,lh,lv,gap,dp
        training_seconds+=time.perf_counter()-tick
        item=dict(epoch=epoch,updates=updates,cumulative_updates=(hist[-1]['cumulative_updates'] if hist else 0)+updates,
            pool_constraints=accepted,pool_proposals=attempted,used_constraints=used,
            unique_constraint_nodes=int(coverage.sum()),rank_loss=losses_total[0]/max(updates,1),
            hom_loss=losses_total[1]/max(updates,1),var_loss=losses_total[2]/max(updates,1),
            gap=float(np.mean(gap_mean)) if gap_mean else None,
            positive_distance=float(np.mean(pos_mean)) if pos_mean else None)
        hist.append(item)
        # With eval_every=refresh_every, this result is reused by next teacher refresh.
        if epoch%args.eval_every==0 or epoch==args.epochs:
            tick=time.perf_counter();ev=evaluate(model,engine,train,args)
            teacher_seconds+=time.perf_counter()-tick;last_eval=epoch
            result=dict(epoch=epoch,**health(ev),
                        clip_fraction=float((ev['score']>=args.clip).mean()))
            rows.append(result)
            print(f'{variant} seed={seed} epoch={epoch} updates={item["cumulative_updates"]} '
                  f'pool={accepted} rank_loss={item["rank_loss"]:.4f} hom_loss={item["hom_loss"]:.4f}',flush=True)
            write_csv(folder/'history.csv',hist);write_csv(folder/'score_history.csv',rows)
    parameter_delta=float((torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])-initial_vector).norm())
    final=dict(seed=seed,variant=variant,graph_mode=engine.mode,parts=engine.nparts,preprocessing=args.preprocessing,
        early_stop=False,selected_epoch=args.epochs,split='8:2',
        parameter_delta_l2=parameter_delta,updates=hist[-1]['cumulative_updates'],seconds=time.perf_counter()-start,
        training_seconds=training_seconds,teacher_eval_seconds=teacher_seconds,sampling_seconds=sampling_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated(engine.device)/1024**3 if engine.device.type=='cuda' else 0,
        peak_reserved_gib=torch.cuda.max_memory_reserved(engine.device)/1024**3 if engine.device.type=='cuda' else 0)
    if args.evaluate_test:
        final.update({'test_'+k:v for k,v in metrics(y,ev['score'],test).items()})
        print(f'Final test: AUROC={final["test_auroc"]:.6f} AUPRC={final["test_auprc"]:.6f}',flush=True)
    np.savez_compressed(folder/'final_scores.npz',score=ev['score'],signed=ev['signed'],center=ev['center'],scale=ev['scale'],train=train,val=val,test=test)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},args=vars(args),
                    input_dim=engine.x.shape[1],variant=variant,seed=seed),folder/'model.pt')
    (folder/'score_health.json').write_text(json.dumps(health(ev),indent=2),encoding='utf-8')
    (folder/'result.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    del optimizer,model;gc.collect()
    return final


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default="datasets")
    p.add_argument('--dataset','--datasets',dest='dataset',default='yelpchi',
                   help='One dataset name or filename (case-insensitive).')
    p.set_defaults(early_stop=False)
    p.add_argument('--output-dir',  default="runs/yelpchi");p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--variants',nargs='+',choices=['homo','energy_only','rank_triplets','var_only'],default=['homo'])
    p.add_argument('--preprocessing',choices=['drop_constant','legacy'],default='drop_constant')
    p.add_argument('--device',default='auto');p.add_argument('--gpu-budget-gb',type=float,default=15.)
    p.add_argument('--graph-mode',choices=['auto','full','bounded'],default='full')
    p.add_argument('--cache-gb',type=float,default=4.)
    p.add_argument('--core-nodes',type=int,default=2048);p.add_argument('--halo-nodes',type=int,default=32768)
    p.add_argument('--halo-edges',type=int,default=4000000);p.add_argument('--fanout',type=int,default=64)
    p.add_argument('--feature-mb',type=float,default=512.);p.add_argument('--partition-seed',type=int,default=17)
    p.add_argument('--transfer-rows',type=int,default=2048)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--steps-per-epoch',type=int,default=1)
    p.add_argument('--refresh-every',type=int,default=5);p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--teacher-mode',choices=['online','initial'],default='online')
    p.add_argument('--hidden-dim',type=int,default=64);p.add_argument('--embedding-dim',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=5e-4)
    p.add_argument('--order',type=int,default=3);p.add_argument('--clip',type=float,default=20.)
    p.add_argument('--mad-floor',type=float,default=1e-8)
    p.add_argument('--pool-size',type=int,default=32768);p.add_argument('--constraints-per-step',type=int,default=8192)
    p.add_argument('--sampling-rounds',type=int,default=12)
    p.add_argument('--rank-gap',type=float,default=.25);p.add_argument('--same-gap',type=float,default=.1)
    p.add_argument('--normal-quantile',type=float,default=.5)
    p.add_argument('--rank-margin',type=float,default=.1);p.add_argument('--hom-margin',type=float,default=.1)
    p.add_argument('--hom-loss',choices=['triplet','huber'],default='triplet')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--lambda-rank',type=float,default=1.);p.add_argument('--lambda-hom',type=float,default=1.)
    p.add_argument('--lambda-var',type=float,default=.05);p.add_argument('--variance-target',type=float,default=1.)
    p.add_argument('--grad-clip',type=float,default=5.);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--evaluate-test',action=argparse.BooleanOptionalAction,default=True,
                   help='Evaluate the test split once after the final epoch.')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    for name in ['epochs','steps_per_epoch','refresh_every','eval_every','hidden_dim','embedding_dim','order',
                 'pool_size','constraints_per_step','sampling_rounds','core_nodes','halo_nodes','halo_edges','fanout','transfer_rows','threads']:
        if getattr(args,name)<1:raise ValueError(f'{name} must be positive')
    for name in ['lr','gpu_budget_gb','feature_mb','clip','mad_floor','temperature','variance_target','grad_clip']:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError(f'Invalid {name}')
    for name in ['rank_gap','same_gap']:
        if not 0<=getattr(args,name)<1:raise ValueError(f'Invalid {name}')
    if not 0<args.normal_quantile<1:raise ValueError('normal_quantile must be (0,1)')
    if min(args.cache_gb,args.weight_decay,args.lambda_rank,args.lambda_hom,args.lambda_var,args.rank_margin,args.hom_margin)<0:
        raise ValueError('Weights, margins and cache budget must be nonnegative')
    if min(args.seeds+[args.partition_seed])<0:raise ValueError('Seeds must be nonnegative')
    torch.set_num_threads(args.threads)
    if args.dataset.lower()=='all':raise ValueError('Run one dataset at a time; specify --dataset NAME.')
    paths=discover(args.data_dir,[args.dataset])
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=False)
    (root/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    results=[]
    for path in paths:
        folder=root/path.name;folder.mkdir()
        x,a,y=load_graph(path,preprocessing=args.preprocessing)
        (folder/'feature_audit.json').write_text(json.dumps(feature_audit(x),indent=2),encoding='utf-8')
        if args.dataset.lower() in ('yelpchi','yelpchi.mat') and len(y)==45954:
            print('Dataset note: 45,954 nodes matches paper YelpChiFull, not the 23,831-node YelpChi. '
                  'Use --dataset YelpChiFull for the full version.',flush=True)
        engine=Engine(x,a,args,folder)
        for seed in args.seeds:
            train,val,test=make_split(y,args.early_stop,seed)
            split_info={name:{'nodes':len(ids),'anomalies':int((y[ids]==1).sum())}
                        for name,ids in [('train',train),('val',val),('test',test)]}
            print(f'{path.name} seed={seed} split={split_info}',flush=True)
            seed_dir=folder/f'seed_{seed}';seed_dir.mkdir()
            (seed_dir/'split_counts.json').write_text(json.dumps(split_info,indent=2),encoding='utf-8')
            torch.manual_seed(seed)
            model=Detector(x.shape[1],args.hidden_dim,args.embedding_dim)
            initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.to(engine.device)
            initial_ev=evaluate(model,engine,train,args)
            np.savez_compressed(seed_dir/'epoch0.npz',score=initial_ev['score'],signed=initial_ev['signed'],train=train,val=val,test=test)
            (seed_dir/'epoch0_health.json').write_text(json.dumps(health(initial_ev),indent=2),encoding='utf-8')
            del model
            for variant in args.variants:
                try:
                    row=run_variant(engine,y,train,val,test,initial,initial_ev,args,seed,variant,seed_dir/variant)
                except torch.cuda.OutOfMemoryError:
                    print('CUDA OOM: no samples were skipped. Rerun in a new folder with '
                          '--graph-mode bounded --gpu-budget-gb 8 --cache-gb 2.',flush=True)
                    raise
                results.append(dict(dataset=path.name,**row));write_csv(root/'results.csv',results)
        engine.clear_cache();del engine,x,a,y;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    summary=[]
    for dataset,variant in sorted({(r['dataset'],r['variant']) for r in results}):
        group=[r for r in results if r['dataset']==dataset and r['variant']==variant]
        row=dict(dataset=dataset,variant=variant,successful_seeds=len(group))
        for metric in ['test_auroc','test_auprc','seconds']:
            values=[r[metric] for r in group if metric in r]
            if values:row.update({metric+'_mean':float(np.mean(values)),metric+'_std':float(np.std(values))})
        summary.append(row)
    write_csv(root/'summary.csv',summary)


if __name__=='__main__':main()
