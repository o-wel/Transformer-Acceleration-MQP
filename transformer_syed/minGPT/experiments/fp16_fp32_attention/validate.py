from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
print('GPU',torch.cuda.get_device_name(),flush=True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p)
                         for p in ['test_flash_attention.py','test_tiled_attention.py','test_tiled_backward.py']])
r=unittest.TextTestRunner(verbosity=2).run(suite)
assert r.wasSuccessful()
print('VALIDATION COMPLETE',flush=True)
