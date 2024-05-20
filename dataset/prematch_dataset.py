import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import resampy

from modules.wavlm.WavLM import WavLM, WavLMConfig
from utils.tools import fast_cosine_dist

DOWNSAMPLE_FACTOR = 320

def make_opensinger_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['ManRaw', 'WomanRaw']
    for f in folders:
        all_files.extend(list((root_path/f).rglob('*.wav')))
    # f.parts[-3][:-3]: Man/Woman
    speakers = [f.parts[-3][:-3] + '-' + f.stem.split('_')[0]  for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df


def main(args):
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else data_root/'wavlm_features'
    device = torch.device(args.device)
    seed = args.seed
    
    ls_df = make_opensinger_df(data_root)

    print(f"Loading wavlm.")
    wavlm_ckpt = torch.load('pretrained/WavLM-Large.pt', map_location='cpu')
    cfg = WavLMConfig(wavlm_ckpt['cfg'])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(wavlm_ckpt['model'])
    wavlm = wavlm.eval()
    wavlm = wavlm.to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    extract(ls_df, wavlm, device, data_root, out_dir)
    print("All done!", flush=True)


@torch.inference_mode()
def get_features(path, wavlm, device='cpu', output_layer=6):
    x, sr = torchaudio.load(path)
    if sr != 16000:
        x = resampy.resample(x.numpy(), sr, 16000, axis=1)
        x = torch.from_numpy(x).to(dtype=torch.float)
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)

    # extract the representation of each layer
    wav_input_16khz = x.to(device)

    features = wavlm.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=False)[0]
    
    return features.squeeze(0)


@torch.inference_mode()
def extract(df: pd.DataFrame, wavlm: nn.Module, device, data_root: Path, out_dir: Path, output_layer=6):
    
    mb = tqdm(df.groupby('speaker'), desc=f'Total Progress')

    for speaker, paths in mb:
        if len(paths) == 1:
            print(f"there is only one audio for speaker {speaker}, ignore him")
            continue
        
        targ_paths = {}
        for i, row in paths.iterrows():
            rel_path = row.path.relative_to(data_root)
            targ_paths[row.path] = (out_dir/rel_path).with_suffix('.pt')
        
        if all([p.exists() for p in targ_paths.values()]):
            continue

        feature_cache = {}
        
        # 1. extract the wavlm features of all the audio of the speaker
        pb = tqdm(paths.iterrows(), total=len(paths), desc=f'extracting {speaker}')
        for i, row in pb:
            feats = get_features(row.path, wavlm, device)
            feature_cache[row.path] = feats

        # 2. replace the wavlm features of each singing audio with the wavlm features of other songs by the same singer.
        pb = tqdm(paths.iterrows(), total=len(paths), desc=f'prematching {speaker}')
        for i, row in pb:
            targ_path = targ_paths[row.path]
            if targ_path.is_file(): continue
            os.makedirs(targ_path.parent, exist_ok=True)

            source_feats = feature_cache[row.path]
            # the audios of the same song are removed since the same song contains repeated phrases.
            song_name = row.path.stem.split('_')[1]
            filtered_matching_feats = {key: value for key, value in feature_cache.items() if song_name not in key.stem}
            matching_pool = list(filtered_matching_feats.values())
            matching_pool = torch.concat(matching_pool, dim=0)
            # calculate the distance and replace each feature with its K neighbors
            matching_pool = matching_pool.to(device)
            dists = fast_cosine_dist(source_feats, matching_pool, device=device)
            best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
            out_feats = matching_pool[best.indices].mean(dim=1) # (N, dim)

            # 3. save pre-matched sequence
            torch.save(out_feats.cpu(), str(targ_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute matched wavlm features for a OpenSinger dataset")

    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--topk', type=int, default=4)

    args = parser.parse_args()
    main(args)