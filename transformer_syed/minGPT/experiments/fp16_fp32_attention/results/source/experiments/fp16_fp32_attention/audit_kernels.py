"""Inspect forward/backward GPU binaries without changing timed measurements."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import triton

OUT=Path(__file__).resolve().parent/'results'


def main():
    tool=Path(triton.__file__).resolve().parent/'backends/nvidia/bin/cuobjdump'
    result={}
    for name in ['forward_training','delta','dq','dkdv']:
        ptx=(OUT/f'{name}.ptx').read_text().lower()
        assert 'mma.sync' not in ptx and '.tf32' not in ptx
        assert 'cvt.f32.f16' not in ptx and 'cvt.rn.f16.f32' not in ptx
        if name!='delta': assert 'fma.rn.f32' in ptx
        sass=subprocess.check_output([str(tool),'--dump-sass',str(OUT/f'{name}.cubin')],text=True)
        resources=subprocess.check_output([str(tool),'--dump-resource-usage',str(OUT/f'{name}.cubin')],text=True)
        (OUT/f'{name}.sass').write_text(sass)
        (OUT/f'{name}-resources.txt').write_text(resources)
        opcodes=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Za-z0-9]+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\s',sass)
        counts=Counter(op.split('.')[0] for op in opcodes)
        assert not any(counts[op] for op in ['HMMA','IMMA','MMA','HGMMA','WGMMA'])
        result[name]=dict(opcode_counts=dict(counts),tensor_mma_instructions=0,
                         hfma2_lines=[s.strip() for s in sass.splitlines() if 'HFMA2' in s],resources=resources)
    (OUT/'INSTRUCTION_AUDIT.json').write_text(json.dumps(result,indent=2))
    print('Forward/backward FP32 PTX and SASS audit passed')


if __name__=='__main__': main()
