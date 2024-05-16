import numpy as np
import resampy
import soundfile as sf

from utils.spectrogram import VoicedAreaDetection

def load_wav(wav_path, sr=24000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # assert fs == sr, f"input audio sample rate must be {sr}Hz. Got {fs}"
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    return wav, fs


def extract_voiced_area(wav_path, hi_freq=1000, hop_size=480, energy_thres=0.5):
    wav, fs = load_wav(wav_path)
    voiced_flag = VoicedAreaDetection(
            x=wav,
            sr=fs,
            n_fft=2048,
            n_shift=hop_size,
            win_length=2048,
            hi_freq=hi_freq,
            energy_thres=energy_thres,
            )
    return voiced_flag

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self