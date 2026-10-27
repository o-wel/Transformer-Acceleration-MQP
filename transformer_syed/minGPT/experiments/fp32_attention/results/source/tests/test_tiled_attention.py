"""IEEE-FP32 prototype: real model layout, offset masks, LoRA, and explicit limits."""
import copy
import unittest
import torch
from mingpt.model import LoRALinear
from test_flash_attention import tiny


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestTiledFP32(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    def test_model_cache_and_lora(self):
        for adapted in (False,True):
            eager=tiny().cuda().eval()
            if adapted:
                eager.enable_lora(2,4)
                for m in eager.modules():
                    if isinstance(m,LoRALinear):
                        torch.nn.init.normal_(m.B.weight,std=.03)
            tiled=copy.deepcopy(eager).set_attention_backend('tiled_fp32')
            x=torch.randint(31,(2,13),device='cuda')
            with torch.inference_mode():
                ref=eager(x)[0]
                cache=None
                pieces=[]
                start=0
                for n in (3,1,5,4):
                    logits,_,cache=tiled(x[:,start:start+n],past_key_values=cache,use_cache=True)
                    pieces.append(logits)
                    start+=n
                torch.testing.assert_close(torch.cat(pieces,1),ref,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(tiled.generate(x,8,use_cache=True),eager.generate(x,8),atol=0,rtol=0)
            with self.assertRaisesRegex(RuntimeError,'inference-only'):
                tiled(x,x)

    def test_direct_reference_and_compiler(self):
        from mingpt.tiled_attention import tiled_attention_fp32
        for width in (8,64,128):
            for nq,past in ((1,0),(37,0),(1,128),(7,128)):
                q=torch.randn(2,3,nq,width,device='cuda')
                k=torch.randn(2,3,nq+past,width,device='cuda')
                v=torch.randn_like(k)
                got,kernel=tiled_attention_fp32(q,k,v,past,return_kernel=True)
                mask=torch.arange(nq+past,device='cuda')[None,:]<=past+torch.arange(nq,device='cuda')[:,None]
                scores=(q.double()@k.double().transpose(-1,-2))*width**-.5
                ref=(scores.masked_fill(~mask,float('-inf')).softmax(-1)@v.double()).float()
                torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-5)
                ptx=kernel.asm['ptx'].lower()
                self.assertNotIn('mma.sync',ptx)
                self.assertNotIn('.tf32',ptx)
                self.assertIn('fma.rn.f32',ptx)
        with self.assertRaises(ValueError):
            tiled_attention_fp32(q.half(),k.half(),v.half(),past)
        with self.assertRaises(ValueError):
            tiled_attention_fp32(q,k,v,0)
