import math
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn

from utils.pitch_extraction import coarse_f0

def load_wav(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            center=center,
            power=1.0,
            norm="slaney",
            n_mels=num_mels,
            mel_scale="slaney",
            f_min=fmin,
            f_max=fmax
        )
        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # print("Padding by", int((n_fft - hop_size)/2), y.shape)
    # pre-padding
    n_pad = hop_size - ( y.shape[1] % hop_size )
    y = F.pad(y.unsqueeze(1), (0, n_pad), mode='reflect').squeeze(1)
    # print("intermediate:", y.shape)

    y = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = spec.abs().clamp_(3e-5)
    # print("Post: ", y.shape, spec.shape)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    train_df = pd.read_csv(a.input_training_file, sep='\t')
    valid_df = pd.read_csv(a.input_validation_file, sep='\t')
    return train_df, valid_df


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, use_alt_melcalc=False):
        self.audio_files = training_files
        if shuffle:
            self.audio_files = self.audio_files.sample(frac=1, random_state=1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.alt_melspec = LogMelSpectrogram(n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        self.use_alt_melcalc = use_alt_melcalc

    def __getitem__(self, index):
        row = self.audio_files.iloc[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(row.audio_path)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.unsqueeze(0)
        seq_len = audio.shape[1]//self.hop_size

        wavlm = torch.load(row.feat_path, map_location='cpu').float() 
        if len(wavlm.shape) == 3:
            wavlm = torch.from_numpy(wavlm)
            wavlm = torch.transpose(wavlm.squeeze(), 0, 1)
        wavlm = torch.repeat_interleave(wavlm, 2, dim=0)
        pitch = np.load(str(row.pitch_path))
        pitch_bins = coarse_f0(pitch)
        pitch = torch.from_numpy(pitch).float()
        pitch_bins = torch.from_numpy(pitch_bins)

        seq_len = min(seq_len, len(pitch), len(wavlm))
        wavlm, pitch, pitch_bins = wavlm[:seq_len,:], pitch[:seq_len], pitch_bins[:seq_len]

        if len(wavlm.shape) < 3:
            wavlm = wavlm.unsqueeze(0) # (1, seq_len, dim)
            pitch = pitch.unsqueeze(0) # (1, seq_len)
            pitch_bins = pitch_bins.unsqueeze(0)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)
            seq_len = frames_per_seg

            if audio.size(1) >= self.segment_size:
                mel_start = random.randint(0, wavlm.size(1) - frames_per_seg)
                wavlm = wavlm[:, mel_start:mel_start + frames_per_seg, :]
                pitch = pitch[:,mel_start:mel_start + frames_per_seg]
                pitch_bins = pitch_bins[:,mel_start:mel_start + frames_per_seg]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            else:
                wavlm = torch.nn.functional.pad(wavlm, (0, 0, 0, frames_per_seg - wavlm.size(1)), 'constant')
                pitch = torch.nn.functional.pad(pitch, (0, frames_per_seg - pitch.size(1)), 'constant')
                pitch_bins = torch.nn.functional.pad(pitch_bins, (0, frames_per_seg - pitch_bins.size(1)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')


        if self.use_alt_melcalc:
            mel_loss = self.alt_melspec(audio[:,:seq_len*self.hop_size])
        else:
            mel_loss = mel_spectrogram(audio[:,:seq_len*self.hop_size], self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        return (wavlm.squeeze(), pitch.squeeze(), pitch_bins.squeeze(), audio.squeeze(0), str(row.audio_path), mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
