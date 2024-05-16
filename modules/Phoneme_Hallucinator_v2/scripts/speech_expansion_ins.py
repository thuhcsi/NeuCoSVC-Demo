import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from utils.hparams import HParams
from models import get_model
import torch


set_size = 200
threshold = 100


def fast_cosine_dist(source_feats, matching_pool):
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


def evaluate(batch, model):
    sample = model.execute(model.sample, batch)
    return sample


def prematch(path, expanded):
    uttrs_from_same_spk = sorted(list(path.parent.rglob('**/*.pt')))
    uttrs_from_same_spk.remove(path)
    candidates = []
    for each in uttrs_from_same_spk:
        candidates.append(torch.load(each))
    candidates = torch.cat(candidates,0)
    candidates = torch.cat([candidates, torch.tensor(expanded)], 0)
    source_feats = torch.load(path)
    source_feats=source_feats.to(torch.float32)
    dists = fast_cosine_dist(source_feats.cpu(), candidates.cpu()).cpu()
    best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
    out_feats = candidates[best.indices].mean(dim=1) # (N, dim)
    return out_feats
    

def single_expand(path, model, num_samples, seed=1234, out_path=None):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # test
    matching_set = torch.load(path, map_location=torch.device('cpu')).numpy()
    matching_set = matching_set / 10
    matching_size = matching_set.shape[0]
    new_samples = []
    cur_num_samples = 0
    while cur_num_samples < num_samples:
        batch = dict()
        if matching_size < threshold:
            num_new_samples = set_size - matching_size
            padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
            batch['b'] = np.concatenate([np.ones_like(matching_set), np.zeros_like(padded_data)], 0)[None, ...]
            batch['x'] = np.concatenate([matching_set, padded_data], axis=0)[None, ...]
            batch['m'] = np.ones_like(batch['b'])
            sample = evaluate(batch, model)
            new_sample = sample[0,matching_size:] * 10
            cur_num_samples += num_new_samples
        else:
            num_new_samples = set_size - threshold
            ind = np.random.choice(matching_size, threshold, replace=False)
            padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
            obs_data = matching_set[ind]
            batch['x'] = np.concatenate([obs_data, padded_data], 0)[None, ...]
            batch['b'] = np.concatenate([np.ones_like(obs_data), np.zeros_like(padded_data)], 0)[None, ...]
            batch['m'] = np.ones_like(batch['b'])
            sample = evaluate(batch, model)
            new_sample = sample[0,num_new_samples:,:] * 10
            cur_num_samples += num_new_samples
        
        new_samples.append(new_sample)
    new_samples = np.concatenate(new_samples, 0)
    new_samples = new_samples[:num_samples]
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, new_samples)
    return new_samples


def single_expand_fast(path):
    # test
    matching_set = torch.load(path).cpu().numpy()
    matching_set = matching_set / 10
    matching_size = matching_set.shape[0]
    batch = dict()
    if matching_size < threshold:
        num_new_samples = set_size - matching_size
    else:
        num_new_samples = set_size - threshold
    batch_size = int(np.ceil(args.num_samples // num_new_samples))
    if matching_size < threshold:
        padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
        batch['b'] = np.concatenate([np.ones_like(matching_set), np.zeros_like(padded_data)], 0)[None, ...]
        batch['x'] = np.concatenate([matching_set, padded_data], axis=0)[None, ...]
        batch['b'] = np.tile(batch['b'], (batch_size, 1, 1))
        batch['x'] = np.tile(batch['b'], (batch_size, 1, 1))
        batch['m'] = np.ones_like(batch['b'])
        sample = evaluate(batch, model)
        new_samples = sample[:,matching_size:, :] * 10
        new_samples = new_samples.reshape((-1, new_samples.shape[-1]))
    else:
        padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
        batch['x'] = []
        for i in range(batch_size):
            ind = np.random.choice(matching_size, threshold, replace=False)
            obs_data = matching_set[ind]
            batch['x'].append(np.concatenate([obs_data, padded_data], 0)[None, ...])
        batch['x'] = np.concatenate(batch['x'], 0)
        batch['b'] = np.concatenate([np.ones_like(obs_data), np.zeros_like(padded_data)], 0)[None, ...]
        batch['b'] = np.tile(batch['b'], (batch_size, 1, 1))
        batch['m'] = np.ones_like(batch['b'])
        sample = evaluate(batch, model)
        new_samples = sample[:,matching_size:, :] * 10
        new_samples = new_samples.reshape((-1, new_samples.shape[-1]))
    new_samples = new_samples[:args.num_samples]
    return new_samples

if __name__ == '__main__':
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--path', type=str, default="matching_set.pt")
    parser.add_argument('--out_path', type=str, default="expanded_set.pt")
    parser.add_argument('--topk', type=int, default=4)
    args = parser.parse_args()
    params = HParams(args.cfg_file)
    # modify config

    t0 = time.time()
    # model
    model = get_model(params)
    model.load()
    t1 = time.time()
    print(f"{t1-t0:.2f}s to load the model")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    path = args.path
    if path.endswith(".pt"):
        t0 = time.time()
        expanded = single_expand(path, model, args.num_samples, args.seed, args.out_path)
        t1 = time.time()
        print(f"{t1-t0:.2f}s to expand the set")
            



