import importlib.util
import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.io import savemat
from graph_io import load_graph,make_split,discover
from bounded_graph import exact_energy_cpu,calibration
from fast_utils import sample_constraints,estimate_gpu_bytes,feature_audit


def graph(n=100):
    rng=np.random.default_rng(2)
    a=(rng.random((n,n))<.08).astype(np.float32)
    a=np.maximum(a,a.T);np.fill_diagonal(a,0)
    return sp.csr_matrix(a)


class CpuTests(unittest.TestCase):
    def test_constant_column_fix_and_legacy_control(self):
        rng=np.random.default_rng(3)
        x=np.column_stack((rng.normal(size=(30,4)),np.full(30,193978.))).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'sample.mat'
            savemat(file,{'Attributes':sp.csr_matrix(x),'Network':graph(30),'Label':np.arange(30)%2})
            legacy,_,_=load_graph(file,'legacy');fixed,_,_=load_graph(file,'drop_constant')
            self.assertEqual(legacy.shape,(30,5));self.assertEqual(fixed.shape,(30,4))
            audit=feature_audit(legacy)
            self.assertGreater(audit['column_energy_fraction'][-1],.999999)
            np.testing.assert_allclose(fixed.toarray(),legacy.toarray()[:,:4],atol=1e-6)

    def test_all_constant_inputs_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'constant.mat'
            savemat(file,{'Attributes':sp.csr_matrix(np.ones((30,2))), 'Network':graph(30),'Label':np.arange(30)%2})
            with self.assertRaisesRegex(ValueError,'All attribute columns'):
                load_graph(file)

    def test_vectorized_sampling_constraints(self):
        a=graph();rank=np.linspace(0,1,100,dtype=np.float32);ids=np.arange(90)
        for kind in ['pairs','triples']:
            p,w,attempts=sample_constraints(ids,a,rank,kind,256,.25,.1,.5,20,np.random.default_rng(4))
            q,_,_=sample_constraints(ids,a,rank,kind,256,.25,.1,.5,20,np.random.default_rng(4))
            np.testing.assert_array_equal(p,q)
            self.assertGreater(len(p),0);self.assertLessEqual(len(p),256)
            self.assertTrue(np.isin(p,ids).all());self.assertTrue((w>0).all())
            if kind=='pairs':self.assertTrue((rank[p[:,0]]-rank[p[:,1]]>=.25).all())
            else:
                aa,pp,nn=p.T
                self.assertTrue((np.asarray(a[aa,pp]).ravel()>0).all())
                self.assertTrue((np.asarray(a[aa,nn]).ravel()==0).all())
                self.assertTrue((np.abs(rank[aa]-rank[pp])<=.1).all())
                self.assertTrue((np.maximum(rank[aa],rank[pp])<=.5).all())
                self.assertTrue((rank[nn]-np.maximum(rank[aa],rank[pp])>=.25).all())
                np.testing.assert_allclose(w,rank[nn]-np.maximum(rank[aa],rank[pp]))

    def test_empty_and_tied_pools(self):
        a=graph();r=np.full(100,.5)
        for kind in ['pairs','triples']:
            for ids in [np.arange(0),np.arange(1),np.arange(100)]:
                p,w,_=sample_constraints(ids,a,r,kind,100,.25,.1,.5,2,np.random.default_rng(4))
                self.assertEqual(len(p),0);self.assertEqual(len(w),0)

    def test_book_split_minority_counts(self):
        y=np.r_[np.zeros(1390,int),np.ones(28,int)]
        train,val,test=make_split(y,True,42)
        self.assertEqual([int(y[ids].sum()) for ids in [train,val,test]],[19,3,6])
        self.assertEqual(len(set(train)&set(val)),0)
        self.assertEqual(len(set(val)&set(test)),0)

    def test_fixed_epoch_split_is_eight_two(self):
        y=np.r_[np.zeros(900,int),np.ones(100,int)]
        train,val,test=make_split(y,False,42)
        self.assertEqual([len(train),len(val),len(test)],[800,0,200])
        self.assertEqual([int(y[train].sum()),int(y[test].sum())],[80,20])
        self.assertEqual(len(np.intersect1d(train,test)),0)
        np.testing.assert_array_equal(np.sort(np.r_[train,test]),np.arange(1000))
        for expected,actual in zip((train,val,test),make_split(y,False,42)):
            np.testing.assert_array_equal(expected,actual)

    def test_unknown_labels_join_training_only(self):
        y=np.r_[np.zeros(90,int),np.ones(10,int),np.full(7,-1,int)]
        train,val,test=make_split(y,False,42)
        self.assertEqual([len(train),len(val),len(test)],[87,0,20])
        self.assertTrue(np.isin(np.arange(100,107),train).all())
        self.assertTrue((y[test]>=0).all())
        self.assertEqual(len(np.intersect1d(train,test)),0)

    def test_extensionless_gadbench_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'weibo').write_bytes(bytes.fromhex('3fa1b446f04f2edd'))
            (root/'YelpChiFull').write_bytes(bytes.fromhex('3fa1b446f04f2edd'))
            (root/'yelpchi.mat').write_bytes(b'placeholder')
            self.assertEqual(discover(root,['WEIBO']),[root/'weibo'])
            self.assertEqual(discover(root,['yelpchi']),[root/'yelpchi.mat'])
            self.assertEqual(discover(root,['YelpChiFull']),[root/'YelpChiFull'])

    def test_calibration_uses_training_reference_and_two_sides(self):
        raw=np.array([[1,10],[2,10],[3,10],[4,10],[0,100]],np.float32)
        ev=calibration(raw,np.arange(4),5,0.5)
        np.testing.assert_allclose(ev['center'],[2.5,10])
        np.testing.assert_allclose(ev['scale'],[1,0.5])
        np.testing.assert_allclose(ev['score'],[1.5,0.5,0.5,1.5,5])
        self.assertLess(ev['signed'][0,0],0)
        self.assertGreater(ev['signed'][3,0],0)

    def test_chebyshev_response_matches_closed_form(self):
        a=sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],np.float32))
        z=np.array([[1,2],[3,4],[2,1]],np.float32)
        op=np.array([[0,-1/np.sqrt(2),0],[-1/np.sqrt(2),0,-1/np.sqrt(2)],
                     [0,-1/np.sqrt(2),0]])
        responses=[op@z,(2*op@op-np.eye(3))@z,(4*op@op@op-3*op)@z]
        expected=np.column_stack([np.log1p((r*r).sum(1)) for r in responses])
        np.testing.assert_allclose(exact_energy_cpu(z,a,3,1),expected,rtol=1e-5,atol=1e-6)

    def test_memory_estimate_monotonic(self):
        small=estimate_gpu_bytes(1000,32,5000,64,32,3)
        large=estimate_gpu_bytes(1000,12047,5000,64,32,3)
        dense=estimate_gpu_bytes(1000,32,500000,64,32,3)
        self.assertGreater(large,small);self.assertGreater(dense,small)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'PyTorch unavailable')
class TorchTests(unittest.TestCase):
    def test_exact_gpu_path_cpu_numerics_and_gradient(self):
        import torch
        from run_fast import parser,Engine,Detector,forward_raw,online_score
        args=parser().parse_args(['--data-dir','.','--output-dir','unused','--device','cpu'])
        a=graph(30);x=np.random.default_rng(4).normal(size=(30,8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            engine=Engine(sp.csr_matrix(x),a,args,Path(tmp))
            model=Detector(8,64,32)
            z=model(torch.tensor(x)).detach().numpy()
            expected=exact_energy_cpu(z,a,3,16)
            _,raw=forward_raw(model,engine.get(0),3)
            np.testing.assert_allclose(raw.detach().numpy(),expected,rtol=2e-5,atol=2e-5)
            ev=calibration(expected,np.arange(30),20,1e-8)
            score=online_score(raw,torch.tensor(ev['center']),torch.tensor(ev['scale']),20)
            np.testing.assert_allclose(score.detach().numpy(),ev['score'],rtol=2e-4,atol=2e-4)
            score.sum().backward()
            self.assertGreater(sum(float(p.grad.norm()) for p in model.parameters()),0)

    def test_public_cli_defaults_and_one_dataset(self):
        import contextlib
        import io
        from run_fast import parser
        args=parser().parse_args([])
        self.assertEqual(args.dataset,'yelpchi')
        self.assertEqual(args.epochs,30)
        self.assertFalse(args.early_stop)
        self.assertEqual(args.graph_mode,'full')
        self.assertFalse(parser().parse_args(['--no-evaluate-test']).evaluate_test)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parser().parse_args(['--dataset','yelpchi','Weibo'])

    def test_triplet_losses_match_paper_formulas(self):
        import torch
        from run_fast import parser,compute_losses
        args=parser().parse_args([])
        z=torch.tensor([[1.,2.],[3.,1.],[2.,4.]])
        score=torch.tensor([.2,.3,.8],requires_grad=True)
        idx=torch.tensor([[0,1,2]])
        rank,hom,var,_,_=compute_losses(z,score,idx,torch.ones(1),
                                       torch.ones(3,dtype=torch.bool),'homo',args)
        expected_rank=.1*np.logaddexp(0.,(.1-.8+.3)/.1)
        expected_hom=.1*np.logaddexp(0.,(.1+abs(.2-.3)-min(abs(.8-.2),abs(.8-.3)))/.1)
        expected_var=np.maximum(0,1-np.sqrt(z.numpy().var(0)+1e-4)).mean()
        self.assertAlmostEqual(float(rank.detach()),expected_rank,places=6)
        self.assertAlmostEqual(float(hom.detach()),expected_hom,places=6)
        self.assertAlmostEqual(float(var.detach()),expected_var,places=6)
        (rank+hom+var).backward()
        self.assertTrue(torch.isfinite(score.grad).all())

    def test_end_to_end_full_and_bounded(self):
        from run_fast import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=np.random.default_rng(8).normal(size=(80,8)).astype(np.float32)
            savemat(root/'toy.mat',{'Attributes':sp.csr_matrix(x),'Network':graph(80),'Label':np.arange(80)%4==0})
            for mode in ['full','bounded']:
                main(['--data-dir',str(root),'--dataset','toy','--output-dir',str(root/mode),
                      '--device','cpu','--graph-mode',mode,'--epochs','2','--eval-every','1','--refresh-every','1',
                      '--pool-size','64','--constraints-per-step','32','--sampling-rounds','2',
                      '--core-nodes','16','--halo-nodes','32','--halo-edges','1000',
                      '--variants','homo','energy_only','rank_triplets','var_only'])
                self.assertTrue((root/mode/'results.csv').exists())
                result_dir=root/mode/'toy.mat'/'seed_42'/'homo'
                self.assertTrue((result_dir/'model.pt').exists())
                result=json.loads((result_dir/'result.json').read_text())
                self.assertFalse(result['early_stop'])
                self.assertEqual(result['selected_epoch'],2)
                self.assertTrue(np.isfinite(result['test_auroc']))
                with np.load(result_dir/'final_scores.npz') as scores:
                    self.assertEqual([len(scores[k]) for k in ['train','val','test']],[64,0,16])
                self.assertFalse((result_dir/'validation.csv').exists())
                self.assertTrue((result_dir/'score_history.csv').exists())


if __name__=='__main__':unittest.main()
