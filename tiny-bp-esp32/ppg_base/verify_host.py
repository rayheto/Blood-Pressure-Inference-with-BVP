"""Compare compiled C++ FP32 inference with the tracked PyTorch model."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from export_firmware import ROOT,fixture_raw,predict
from model import TinyBP


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,required=True,help='Compiled firmware/tools/host_smoke.cpp')
    parser.add_argument('--raw',type=Path,action='append',default=[],
                        help='Optional 125 Hz continuous raw PPG .npy for additional windows')
    args=parser.parse_args()
    model=TinyBP().eval()
    model.load_state_dict(torch.load(ROOT/'model/best.pt',map_location='cpu',weights_only=True))
    with np.load(ROOT/'ppg_base/selected_head.npz',allow_pickle=False) as d:
        head={k:d[k] for k in d.files}
    preprocess=json.loads((ROOT/'model/preprocess.json').read_text(encoding='utf-8'))
    windows=[('fixture',fixture_raw((ROOT/'firmware/main/ppg_fixture.hpp').read_text(encoding='utf-8')))]
    for path in args.raw:
        raw=np.load(path,mmap_mode='r')
        for seconds in (120,300,600):
            end=seconds*125
            if len(raw)>=end:
                windows.append((f'{path.stem}@{seconds}s',np.asarray(raw[end-1250:end],np.float32)))
    worst=np.zeros(2)
    for name,raw in windows:
        expected=predict(model,head,raw,preprocess)
        payload='\n'.join(format(float(v),'.9g') for v in raw)+'\n'
        proc=subprocess.run([str(args.exe),'--stdin'],input=payload,text=True,capture_output=True,check=True)
        actual=np.asarray([float(v) for v in proc.stdout.split()],np.float32)
        error=np.abs(actual-expected)
        worst=np.maximum(worst,error)
        print(name,'torch',expected.round(5).tolist(),'C++',actual.round(5).tolist(),
              'abs_error',error.round(6).tolist())
    if (worst>.05).any():raise SystemExit(f'FP32 parity failed: {worst}')
    print('PASS',len(windows),'windows; max abs error mmHg',worst.tolist())

if __name__=='__main__':main()
