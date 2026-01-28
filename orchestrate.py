# orchestrate.py
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(p.stdout.strip())
    if p.stderr: print(p.stderr.strip())
    return p.stdout.strip().splitlines()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sep", default="\\t")
    ap.add_argument("--target", required=True)
    ap.add_argument("--resample", default="30S")
    ap.add_argument("--order", nargs=3, type=int, default=[1,0,1])
    ap.add_argument("--seasonal", nargs=4, type=int, default=[0,0,0,0])
    ap.add_argument("--h", type=int, default=30)
    ap.add_argument("--processed_dir", default="processed")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--forecasts_dir", default="forecasts")
    args = ap.parse_args()

    # 1) PREP
    out_paths = run([sys.executable,"sarimax_prep.py","--input",args.input,"--sep",args.sep,
                     "--target",args.target,"--resample",args.resample,"--outdir",args.processed_dir])
    data_path, meta_path = out_paths[0], out_paths[1]

    # 2) TRAIN
    outs_train = run([sys.executable,"sarimax_train.py","--data",data_path,"--order",
                      *map(str,args.order),"--seasonal",*map(str,args.seasonal),
                      "--outdir",args.models_dir])
    model_path = outs_train[0]

    # 3) FORECAST
    run([sys.executable,"sarimax_forecast.py","--model",model_path,"--data",data_path,
         "--meta",meta_path,"--h",str(args.h),"--outdir",args.forecasts_dir])

if __name__ == "__main__":
    main()
