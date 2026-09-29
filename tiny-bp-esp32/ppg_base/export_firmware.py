"""Export the selected current-PPG head plus TinyBP CNN as FP32 firmware arrays."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import TinyBP  # noqa: E402

MODEL_TAG='ppg-current-seed20260929-fp32'


def c_float(value):
    token=format(float(np.float32(value)),'.9g')
    if '.' not in token and 'e' not in token:token+='.0'
    return token+'f'


def array(name,value):
    flat=np.asarray(value,dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all():raise ValueError(name+' has nonfinite values')
    lines=[f'inline constexpr float {name}[{len(flat)}] = {{']
    for start in range(0,len(flat),8):
        lines.append('    '+', '.join(c_float(v) for v in flat[start:start+8])+',')
    return '\n'.join(lines+['};',''])


def fixture_raw(header):
    match=re.search(r'kRawPpg\[\d+\]\s*=\s*\{(.*?)\};',header,re.S)
    if not match:raise ValueError('fixture array missing')
    raw=np.asarray([float(token.strip().removesuffix('f')) for token in match.group(1).split(',')
                    if token.strip()],dtype=np.float32)
    if raw.shape!=(1250,):raise ValueError(f'fixture shape {raw.shape}')
    return raw


def predict(model,head,raw,preprocess):
    x=torch.from_numpy(((raw-preprocess['mean'])/preprocess['std'])[None,None,:].astype(np.float32))
    with torch.no_grad():
        latent=model.pool(model.features(x)).flatten(1)[0]
        tiny=model.head(latent)*100
        mean=torch.from_numpy(head['mean'][:34]);std=torch.from_numpy(head['std'][:34])
        current=torch.cat((latent,tiny/100))
        z=torch.cat(((current-mean)/std,torch.ones(1)))
        h=torch.relu(torch.from_numpy(head['fc0_weight'])@z+torch.from_numpy(head['fc0_bias']))
        residual=torch.from_numpy(head['fc1_weight'])@h+torch.from_numpy(head['fc1_bias'])
        return (tiny+torch.tanh(residual)*torch.tensor([40.,25.])).numpy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--head',type=Path,default=ROOT/'ppg_base/selected_head.npz')
    parser.add_argument('--base',type=Path,default=ROOT/'model/best.pt')
    parser.add_argument('--firmware',type=Path,default=ROOT/'firmware/main')
    args=parser.parse_args()
    head={k:v for k,v in np.load(args.head,allow_pickle=False).items()}
    expected={'fc0_weight':(32,35),'fc0_bias':(32,),
              'fc1_weight':(2,32),'fc1_bias':(2,), 'mean':(34,),'std':(34,)}
    for k,shape in expected.items():
        if k not in head or head[k].shape!=shape:raise ValueError(f'{k} must have shape {shape}')
    model=TinyBP().eval()
    model.load_state_dict(torch.load(args.base,map_location='cpu',weights_only=True))
    preprocess=json.loads((ROOT/'model/preprocess.json').read_text(encoding='utf-8'))
    sections=['#pragma once','#include <cstddef>','namespace bp_weights {',
              f'inline constexpr const char* kModelTag = "{MODEL_TAG}";']
    for index in range(4):
        conv=model.features[index*3]
        bn=model.features[index*3+1]
        scale=(bn.weight/torch.sqrt(bn.running_var+bn.eps)).detach().numpy()
        weight=conv.weight.detach().numpy()*scale[:,None,None]
        bias=(bn.bias-bn.running_mean*bn.weight/torch.sqrt(bn.running_var+bn.eps)).detach().numpy()
        sections.extend((array(f'kConv{index}Weight',weight),array(f'kConv{index}Bias',bias)))
    sections.extend((array('kTinyHeadWeight',model.head.weight.detach().numpy()),
                     array('kTinyHeadBias',model.head.bias.detach().numpy())))
    for key,name in [('mean','kFeatureMean'),('std','kFeatureStd'),
                     ('fc0_weight','kCorrection0Weight'),('fc0_bias','kCorrection0Bias'),
                     ('fc1_weight','kCorrection1Weight'),('fc1_bias','kCorrection1Bias')]:
        sections.append(array(name,head[key]))
    sections.extend(('}  // namespace bp_weights',''))
    args.firmware.mkdir(parents=True,exist_ok=True)
    weights_path=args.firmware/'ppg_base_weights.hpp'
    weights_path.write_text('\n'.join(sections),encoding='utf-8')
    header_path=args.firmware/'ppg_fixture.hpp'
    original=header_path.read_text(encoding='utf-8')
    raw=fixture_raw(original)
    estimate=predict(model,head,raw,preprocess)
    changed=original
    for key,value in [('kExpectedFloatSbp',estimate[0]),('kExpectedFloatDbp',estimate[1])]:
        changed,n=re.subn(rf'({key}\s*=\s*)[^;]+;',lambda m:m.group(1)+c_float(value)+';',changed)
        if n!=1:raise ValueError(key+' missing from fixture')
    header_path.write_text(changed,encoding='utf-8')
    meta_path=args.firmware/'ppg_fixture.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['raw_ppg_sha256']!=hashlib.sha256(raw.tobytes()).hexdigest():
        raise ValueError('fixture SHA mismatch')
    meta['model']=MODEL_TAG
    meta['float_model_sbp_dbp_mmhg']=[float(v) for v in estimate]
    meta_path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    preprocess_header='\n'.join(['#pragma once','#include <cstddef>','namespace bp_config {',
        'inline constexpr std::size_t kSampleRateHz = 125;',
        'inline constexpr std::size_t kWindowSamples = 1250;',
        'inline constexpr std::size_t kStrideSamples = kSampleRateHz;',
        f'inline constexpr float kPpgMean = {c_float(preprocess["mean"])};',
        f'inline constexpr float kPpgStd = {c_float(preprocess["std"])};',
        '}  // namespace bp_config',''])
    (args.firmware/'ppg_preprocess.hpp').write_text(preprocess_header,encoding='utf-8')
    print(json.dumps({'model':MODEL_TAG,'fixture_prediction_mmhg':estimate.tolist(),
                      'weights_bytes':weights_path.stat().st_size},indent=2))

if __name__=='__main__':main()
