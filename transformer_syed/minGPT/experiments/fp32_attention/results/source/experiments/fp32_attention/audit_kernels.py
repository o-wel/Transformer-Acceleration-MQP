"""Audit saved Triton cubins; runnable after the GPU measurements finish."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import triton

OUT=Path(__file__).resolve().parent/'results'


def main():
    cuobjdump=Path(triton.__file__).resolve().parent/'backends/nvidia/bin/cuobjdump'
    audit={}
    for phase in ['prefill','decode']:
        cubin=OUT/f'tiled-{phase}.cubin'
        sass=subprocess.check_output([str(cuobjdump),'--dump-sass',str(cubin)],text=True)
        resources=subprocess.check_output([str(cuobjdump),'--dump-resource-usage',str(cubin)],text=True)
        (OUT/f'tiled-{phase}.sass').write_text(sass)
        (OUT/f'tiled-{phase}-resources.txt').write_text(resources)
        opcodes=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Za-z0-9]+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\s',sass)
        counts=Counter(op.split('.')[0] for op in opcodes)
        assert counts['FFMA']>0
        assert not any(counts[op] for op in ['HMMA','IMMA','MMA','HGMMA','WGMMA'])
        half_lines=[line.strip() for line in sass.splitlines() if 'HFMA2' in line]
        assert all('-RZ, RZ,' in line for line in half_lines)
        ptx=(OUT/f'tiled-{phase}.ptx').read_text().lower()
        assert 'fma.rn.f32' in ptx and 'mma.sync' not in ptx and '.tf32' not in ptx
        audit[phase]=dict(opcode_counts=dict(counts),tensor_mma_instructions=0,
                         constant_materialization_note='Zero-input HFMA2.MMA materializes constants, not Q/K/V arithmetic or Tensor Core matrix products',
                         hfma2_constant_lines=half_lines)
    (OUT/'INSTRUCTION_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print('PTX/SASS audit passed for prefill and decode')


if __name__=='__main__':
    main()
