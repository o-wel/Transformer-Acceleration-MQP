"""CLI used exclusively from the frozen source snapshot."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.paper_hyperparameters.common import save,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','search-train','search-runtime','select','confirm-train','confirm-runtime','quality','report'])
    p.add_argument('--study',type=Path,required=True);p.add_argument('--index',type=int,default=0);args=p.parse_args()
    study=args.study.resolve();verify(study)
    try:
        if args.stage=='preflight':
            from experiments.paper_hyperparameters.preflight import run
            run(study)
        elif args.stage in ('search-train','confirm-train'):
            from experiments.paper_hyperparameters.train import run_search,run_confirmation
            (run_search if args.stage=='search-train' else run_confirmation)(study,args.index)
        elif args.stage in ('search-runtime','confirm-runtime','quality'):
            from experiments.paper_hyperparameters.runtime import search_runtime,confirmation_runtime,confirmation_quality
            {'search-runtime':search_runtime,'confirm-runtime':confirmation_runtime,'quality':confirmation_quality}[args.stage](study,args.index)
        else:
            from experiments.paper_hyperparameters.analysis import select,report
            (select if args.stage=='select' else report)(study)
    except Exception as error:
        import traceback
        save(study/'failures'/f'{args.stage}-{args.index}.json',dict(stage=args.stage,index=args.index,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
