import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor

from .models import GeneratorNSF as Generator, PitchEncoder as PitchEmbedding
from .wavlm.WavLM import WavLM, WavLMConfig
from utils.tools import AttrDict, load_wav


SPEAKER_INFORMATION_WEIGHTS = [
    0, 0, 0, 0, 0, 0,  # layer 0-5
    1.0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, # layer 15
    0, 0, 0, 0, 0, 0, # layer 16-21
    0, # layer 22 
    0, 0 # layer 23-24
]
SPEAKER_INFORMATION_LAYER = 6


def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


class SVCNN(nn.Module):
    def __init__(self,
        model_ckpt_path,
        model_cfg_path='configs/config.json',
        wavlm_ckpt_path='pretrained/WavLM-Large.pt',
        device='cpu'
    ) -> None:
        super().__init__()
        # set which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
        # load model
        with open(model_cfg_path) as f:
            data = f.read()
        json_config = json.loads(data)
        model_cfg = AttrDict(json_config)
        pitch_emb = PitchEmbedding(model_cfg).to(device)
        model = Generator(model_cfg).to(device)
        state_dict_g = torch.load(model_ckpt_path, map_location='cpu')
        pitch_emb.load_state_dict(state_dict_g['pitch_encoder'])
        model.load_state_dict(state_dict_g['generator'])
        model.remove_weight_norm()
        self.pitch_emb = pitch_emb.to(device).eval()
        self.model = model.to(device).eval()
        print(f"Generator loaded with {sum([p.numel() for p in model.parameters() if p.requires_grad]) + sum([p.numel() for p in pitch_emb.parameters() if p.requires_grad]):,d} parameters.")
        self.h = model_cfg
        # load wavlm
        wavlm_ckpt = torch.load(wavlm_ckpt_path, map_location='cpu')
        cfg = WavLMConfig(wavlm_ckpt['cfg'])
        wavlm = WavLM(cfg)
        wavlm.load_state_dict(wavlm_ckpt['model'])
        wavlm.to(device)
        self.wavlm = wavlm.eval()
        print('wavlm loaded')
        self.device = torch.device(device)
        self.sr = 16000
        self.hop_length = 320

    def get_matching_set(self, p: Path|str, weights=None, vad_trigger_level=0, out_path=None) -> Tensor:
        """ Get concatenated wavlm features for the matching set using all waveforms in `wavs`, 
        specified as either a list of paths or list of loaded waveform tensors of 
        shape (channels, T), assumed to be of 16kHz sample rate.
        Optionally specify custom WavLM feature weighting with `weights`.
        """
        feats = []
        # 只取有声段作为matching_set

        # x, sr = torchaudio.load(p, normalize=True)
        x, sr = load_wav(p, self.sr)
        # x, _, __ = trim_long_silences(x, sr)
        audio_length = len(x)
        slice_length = 60*self.sr

        for start_pos in range(0, audio_length, slice_length):
            end_pos = start_pos + slice_length
            slice_x = x[start_pos:end_pos]
            slice_x = torch.from_numpy(slice_x).float()
            feats.append(self.get_features(slice_x, weights=self.weighting if weights is None else weights, vad_trigger_level=vad_trigger_level))
            
        feats = torch.concat(feats, dim=0).cpu()
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(feats, out_path)
        return feats
        

    @torch.inference_mode()
    def vocode(self, c: Tensor, pitch:Tensor) -> Tensor:
        y_g_hat = self.model(c, pitch)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat


    @torch.inference_mode()
    def get_features(self, path, weights=None, vad_trigger_level=0):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if weights == None: weights = self.weighting
        if type(path) in [str, Path]:
            # x, sr = torchaudio.load(path, normalize=True)
            x, sr = load_wav(path, self.sr)
            x = torch.from_numpy(x).float()
        else:
            x: Tensor = path
            sr = self.sr
        if x.dim() == 1: x = x[None]
        assert sr == self.sr, f"input audio sample rate must be 16kHz. Got {sr}"
        
        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim, sr = apply_effects_tensor(
                waveform_reversed_front_trim, sr, [["reverse"]]
            )
            x = waveform_end_trim

        # extract the representation of each layer
        wavs_split = torch.tensor_split(x, (x.shape[1]-1)//(sr*30)+1, dim=1)
        feature_list = []
        for wav_chunk in wavs_split:
            wav_input_16khz = wav_chunk.to(self.device)
            if torch.allclose(weights, self.weighting):
                # use fastpath
                features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
                features = features.squeeze(0)
                feature_list.append(features)
            else:
                # use slower weighted
                rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
                features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
                # save full sequence
                features = (features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
                feature_list.append(features)
        
        return torch.cat(feature_list)


    @torch.inference_mode()
    def match(self, query_seq: Tensor, pitch:Tensor, pitch_bins:Tensor, synth_set: Tensor,
              topk: int = 4, query_mask: Tensor = None, alpha = 0, tgt_loudness_db: float | None = -16,
              target_duration: float | None = None, device: str | None = None) -> Tensor:
        """ Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
        with k=`topk`. Inputs:
            - `query_seq`: Tensor (N1, dim) of the input/source query features.
            - `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
            - `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
                vector to a vector in the matching set, and then use the corresponding vector from the synth set during decoder synthesis.
                By default, and for best performance, this should be identical to the matching set. 
            - `topk`: k in the kNN -- the number of nearest neighbors to average over.
            - `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable. 
            - `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
            - `device`: if None, uses default device at initialization. Otherwise uses specified device
        Returns:
            - converted waveform of shape (T,)
        """
        device = torch.device(device) if device is not None else self.device
        synth_set = synth_set.to(device)
        query_seq = query_seq.to(device)
        pitch = pitch.to(device)
        pitch_bins = pitch_bins.to(device)

        if target_duration is not None:
            target_samples = int(target_duration*self.sr)
            scale_factor = (target_samples/self.hop_length) / query_seq.shape[0] # n_targ_feats / n_input_feats
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

        dists = fast_cosine_dist(query_seq, synth_set, device=device)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = (1-alpha) * synth_set[best.indices].mean(dim=1) + alpha *  query_seq
        if query_mask is not None:
            query_mask = query_mask[..., None].repeat([1, out_feats.shape[-1]])
            out_feats = out_feats * query_mask + query_seq * (query_mask == False)
        out_feats = torch.repeat_interleave(out_feats, 2, 0)
        out_feats = self.pitch_emb(out_feats, pitch_bins)
        
        prediction = self.vocode(out_feats[None].to(device), pitch.unsqueeze(0)).cpu().squeeze()
        
        # normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
        else: pred_wav = prediction
        return pred_wav