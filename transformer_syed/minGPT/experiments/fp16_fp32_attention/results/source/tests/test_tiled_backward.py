"""First-order FP32 backward accuracy, training parity, layout and memory contract."""
import copy
import unittest
import torch
from mingpt.model import LoRALinear
from test_flash_attention import tiny


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestTiledBackward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    def test_float64_reference_and_noncontiguous_grad(self):
        from mingpt.tiled_attention import tiled_attention_fp32
        for b,h,n,d,past in [(2,3,37,8,0),(1,2,33,64,0),(1,2,7,64,128),
                              (1,2,1,128,127),(1,1,65,128,0)]:
            inputs=[]
            for length in [n,n+past,n+past]:
                x=(.3*torch.randn(b,length,h,d,device='cuda')).transpose(1,2).detach().requires_grad_(True)
                inputs.append(x)
            q,k,v=inputs
            go=torch.randn(b,n,h,d,device='cuda').transpose(1,2)
            out=tiled_attention_fp32(q,k,v,past)
            grads=torch.autograd.grad(out,inputs,go)
            double=[x.detach().double().requires_grad_(True) for x in inputs]
            mask=torch.arange(n+past,device='cuda')[None,:]<=past+torch.arange(n,device='cuda')[:,None]
            scores=double[0]@double[1].transpose(-1,-2)*d**-.5
            ref=scores.masked_fill(~mask,float('-inf')).softmax(-1)@double[2]
            expected=torch.autograd.grad(ref,double,go.double())
            torch.testing.assert_close(out,ref.float(),atol=2e-5,rtol=2e-5)
            for got,want in zip(grads,expected):
                torch.testing.assert_close(got,want.float(),atol=3e-5,rtol=3e-4)

    def test_model_gradients_and_optimizer_steps(self):
        for adapted in [False,True]:
            eager=tiny().cuda().train()
            if adapted:
                eager.enable_lora(2,4)
                for m in eager.modules():
                    if isinstance(m,LoRALinear):
                        torch.nn.init.normal_(m.B.weight,std=.03)
            tiled=copy.deepcopy(eager).set_attention_backend('tiled_fp32')
            opts=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
                  for m in [eager,tiled]]
            for step in range(4):
                x=torch.randint(31,(2,13),device='cuda')
                losses=[]
                for model,opt in zip([eager,tiled],opts):
                    opt.zero_grad(set_to_none=True)
                    loss=model(x,x.roll(-1,1))[1]
                    loss.backward()
                    losses.append(loss.detach())
                torch.testing.assert_close(losses[0],losses[1],atol=2e-5,rtol=2e-5)
                for (name,p),(_,q) in zip(eager.named_parameters(),tiled.named_parameters()):
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4,msg=name)
                    else:
                        self.assertIsNone(q.grad)
                for opt in opts:
                    opt.step()
            for (name,p),(_,q) in zip(eager.named_parameters(),tiled.named_parameters()):
                torch.testing.assert_close(p,q,atol=2e-5,rtol=2e-4,msg=name)

    def test_linear_saved_state_and_no_dropout(self):
        from mingpt.tiled_attention import tiled_attention_fp32
        saved=[]
        def pack(t):
            saved.append(tuple(t.shape))
            return t
        q=torch.randn(2,3,65,64,device='cuda',requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
            out=tiled_attention_fp32(q,q,q)
        self.assertFalse(any(len(s)>=2 and s[-2:]==(65,65) for s in saved))
        # Q/K/V + O + one scalar log-normalizer per query, not an NxN buffer.
        self.assertEqual(sum(__import__('math').prod(s) for s in saved),4*q.numel()+2*3*65)
        out.square().mean().backward()
        model=tiny().cuda().set_attention_backend('tiled_fp32').train()
        model.transformer.h[0].attn.attn_dropout.p=.1
        with self.assertRaisesRegex(RuntimeError,'attention dropout'):
            model(torch.ones(1,4,dtype=torch.long,device='cuda'))

    def test_backward_ptx(self):
        from mingpt.tiled_attention import _forward,_run_backward
        q=torch.randn(1,2,37,64,device='cuda')
        out,lse,fwd=_forward(q,q,q,0,True)
        _,kernels=_run_backward(q,q,q,out,lse,torch.randn_like(out),0,True)
        for name,kernel in {'forward':fwd,**kernels}.items():
            ptx=kernel.asm['ptx'].lower()
            self.assertNotIn('mma.sync',ptx,name)
            self.assertNotIn('.tf32',ptx,name)
            if name!='delta':
                self.assertIn('fma.rn.f32',ptx,name)
