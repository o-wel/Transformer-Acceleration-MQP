import json
from pathlib import Path
import subprocess
import sys
root=Path(sys.argv[1]).resolve()
jobs=json.loads((root/'jobs.json').read_text())
print(subprocess.check_output(['sacct','-j',','.join(r['job'] for r in jobs),'--format=JobID,State,ExitCode,Elapsed','-P'],text=True))
for run in sorted(root.glob('*_fp*')):
    counts={name:sum(len(json.loads(f.read_text())) for f in run.glob(pattern)) for name,pattern in [('calibration','calibration-*.json'),('confirmation','confirmation-*.json'),('quality','quality-*.json')]}
    print(run.name,counts)
