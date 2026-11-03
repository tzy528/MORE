"""NumPy/SciPy utilities. Candidate construction never receives labels."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def estimate_gpu_bytes(n, d, nnz, hidden, embedding, order):
    # Conservative planning estimate, not a CUDA allocator hard limit.
    features = 4*n*d
    sparse = 20*nnz + 8*(n+1)  # COO storage; CSR would be smaller.
    activation = 4*n*(8*hidden+(6*order+12)*embedding)
    parameters = 24*(d*hidden+hidden*embedding+hidden+embedding)
    return int(1.5*features + 3*sparse + activation + parameters + 512*1024**2)


def normalized_operator(a):
    degree = np.asarray(a.sum(1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inv, where=degree>0)
    return -(sp.diags(inv) @ a @ sp.diags(inv)).tocsr()


def sample_constraints(ids, a, rank, kind, count, gap, same_gap, normal_quantile,
                       rounds, rng):
    """Vectorized finite-budget rejection sampling; no Python per-triplet loop.

    Uniform outgoing-edge proposals from eligible sources, then destination
    eligibility, energy conditions and original-graph non-neighbor rejection.
    Returns global node IDs and energy confidence weights.
    """
    ids=np.sort(np.asarray(ids,np.int64))
    width=2 if kind=='pairs' else 3
    empty=np.empty((0,width),np.int64)
    if len(ids)<width or count==0:
        return empty,np.empty(0,np.float32),0
    parts=[]; weights=[]; total=0; attempted=0
    eligible=np.zeros(a.shape[0],bool); eligible[ids]=True
    if kind=='triples':
        prefix=np.cumsum(np.diff(a.indptr)[ids],dtype=np.int64)
        if prefix[-1]==0:
            return empty,np.empty(0,np.float32),0
    for _ in range(rounds):
        need=count-total
        if need<=0:break
        size=min(131072,max(1024,8*need))
        attempted+=size
        if kind=='pairs':
            u=ids[rng.integers(len(ids),size=size)]
            v=ids[rng.integers(len(ids),size=size)]
            delta=rank[u]-rank[v]
            mask=(np.abs(delta)>=gap)&(delta!=0)&(u!=v)
            u,v,delta=u[mask],v[mask],delta[mask]
            part=np.column_stack((np.where(delta>0,u,v),np.where(delta>0,v,u)))
            w=np.abs(delta)
        else:
            draw=rng.integers(prefix[-1],size=size)
            at=np.searchsorted(prefix,draw,side='right')
            before=np.where(at>0,prefix[np.maximum(at-1,0)],0)
            u=ids[at]
            p=a.indices[a.indptr[u]+draw-before]
            n=ids[rng.integers(len(ids),size=size)]
            pos=np.abs(rank[u]-rank[p]); neg=rank[n]-np.maximum(rank[u],rank[p])
            mask=(eligible[p] & (u!=n) & (p!=n) &
                  (np.maximum(rank[u],rank[p])<=normal_quantile) & (pos<=same_gap) & (neg>=gap))
            u,p,n,pos,neg=u[mask],p[mask],n[mask],pos[mask],neg[mask]
            if len(u):
                nonneighbor=np.asarray(a[u,n]).ravel()==0
                u,p,n,pos,neg=u[nonneighbor],p[nonneighbor],n[nonneighbor],pos[nonneighbor],neg[nonneighbor]
            part=np.column_stack((u,p,n))
            # Eq. (13): q_n - max(q_a, q_p).
            w=neg
        parts.append(part[:need]); weights.append(w[:need]); total+=min(len(part),need)
    if not parts:
        return empty,np.empty(0,np.float32),attempted
    return np.concatenate(parts).astype(np.int64),np.concatenate(weights).astype(np.float32),attempted


def feature_audit(x):
    if sp.issparse(x):
        from sklearn.utils.sparsefuncs import mean_variance_axis
        mean,var=mean_variance_axis(x.astype(np.float64),axis=0)
    else:
        mean=np.mean(x,axis=0,dtype=np.float64);var=np.var(x,axis=0,dtype=np.float64)
    energy=mean**2+var
    total=float(energy.sum())
    return dict(mean=mean.tolist(),variance=var.tolist(),constant_columns=np.flatnonzero(var==0).tolist(),
                column_energy_fraction=(energy/total if total else np.zeros_like(energy)).tolist(),
                mean_offset_energy_fraction=float((mean**2).sum()/total) if total else None)
