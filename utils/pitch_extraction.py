import numpy as np
import os
import random
import librosa
import parselmouth

from utils.spectrogram import load_wav

np.random.seed(0)
random.seed(0)


def REAPER_F0(wav_path, sr=24000, frame_period=0.01):  # frame_period s
    if not os.path.isfile(f'{wav_path}.f0'):
        cmd = f'path-to-reaper -i {wav_path} -f {wav_path}.f0 -e {frame_period} -x 1000 -m 65 -a'
        os.system(cmd)
    f0 = []
    try:
        with open(f'{wav_path}.f0', 'r') as rf:
            for line in rf.readlines()[7:]:
                f0.append(float(line.split()[2]))
    except FileNotFoundError as e:
        return None

    cmd = f'rm -f {wav_path}.f0'
    os.system(cmd)

    f0 = np.array(f0)
    minus_one_indexes = (f0 == -1)
    f0[minus_one_indexes] = 0

    return f0


def ParselMouth_F0(wav, sr=24000, frame_period=0.01):
    wav = parselmouth.Sound(wav, sampling_frequency=sr)
    pitch = wav.to_pitch(time_step=frame_period, pitch_floor=65, pitch_ceiling=1000)
    f0 = pitch.selected_array['frequency']

    return f0


def PYIN_F0(wav, sr=24000, frame_period=10):
    fmin = librosa.note_to_hz('C2')  # ~65Hz
    fmax = librosa.note_to_hz('C7')  # ~2093Hz
    # fmax = fs/2
    f0, voiced_flag, voiced_prob = librosa.pyin(
        wav, fmin=fmin, fmax=fmax, sr=sr, frame_length=int(sr*frame_period/1000*4))
    f0 = np.where(np.isnan(f0), 0.0, f0)
    return f0


def pad_arrays(arrays: list[np.ndarray], std_len: int):
    """
    Pad arrays value to a specified standard length.

    Args:
        arrays (List[numpy.ndarray]): List of arrays to be padded.
        std_len (int): Standard length to which the arrays will be padded.

    Returns:
        List[numpy.ndarray]: List of padded arrays.

    Raises:
        ValueError: If the length of any array in the input list is greater than the specified standard length.
    
    """
    padded_arrays = []
    for arr in arrays:
        cur_len = len(arr)
        if cur_len <= std_len:
            pad_width = std_len - cur_len
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            padded_arr = np.pad(arr, (left_pad, right_pad), 'edge')
            padded_arrays.append(padded_arr)
        else:
            raise ValueError(f'cur_len: {cur_len}, std_len: {std_len}.')
    return padded_arrays


def compute_pitch(wav_path: str, pitch_path: str=None, frame_period=0.01):
    """
    Computes the pitch information from an audio waveform.

    Args:
        wav_path (str): Path to the audio waveform file (must be 24kHz).
        pitch_path (str, optional): Path to save or load the computed pitch information as a numpy file. 
            If specified, the function will first attempt to load the pitch information from this path. 
            If the file does not exist, the pitch will be computed and saved to this path. 
            Defaults to None.
        frame_period (float, optional): Time duration in seconds for each frame. Defaults to 0.01.

    Returns:
        numpy.ndarray: Computed pitch information.

    Notes:
        For precise pitch representation, the pitch values are extracted by the median of three methods: 
        the PYIN, the REAPER, and the Parselmouth.

    """
    if pitch_path is not None and os.path.isfile(pitch_path):
        pitch = np.load(pitch_path)
        return pitch
    else:
        # extract pitch using 24kHz audio
        wav, fs = load_wav(wav_path, 24000)
        f0_std_len = wav.shape[0] // int(frame_period*fs) + 1

        compute_median = []

        # Compute pitch using PYIN algorithm
        f0 = PYIN_F0(wav, sr=fs, frame_period=frame_period*1000)
        compute_median.append(f0)

        # Compute pitch using ParselMouth algorithm
        f0 = ParselMouth_F0(wav, sr=fs, frame_period=frame_period)
        compute_median.append(f0)

        # Compute pitch using REAPER algorithm
        f0 = REAPER_F0(wav_path, sr=fs, frame_period=frame_period)

        if f0 is not None:
            compute_median.append(f0)

        # Compute median F0
        compute_median = pad_arrays(compute_median, f0_std_len)
        compute_median = np.array(compute_median)
        median_f0 = np.median(compute_median, axis=0)
        if pitch_path is not None:
            os.makedirs(pitch_path.parent, exist_ok=True)
            np.save(pitch_path, median_f0)
        return median_f0


def coarse_f0(f0):
    """
    Converts the fundamental frequency (f0) values to coarse f0 values.

    Args:
        f0 (ndarray): Array of fundamental frequency values.

    Returns:
        ndarray: Array of coarse f0 values.

    """
    f0_bin = 256
    f0_max = 1000.0
    f0_min = 65.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
        f0_bin - 2
    ) / (f0_mel_max - f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


def extract_pitch_ref(wav_path: str, ref_path: str, predefined_factor=0, speech_enroll=False):
    """
    Extracts pitch information from an audio waveform and adjusts it based on a reference audio.

    Args:
        wav_path (str): Path to the audio waveform file.
        ref_path (str): Path to the reference audio waveform file.
        predefined_factor (float, optional): Predefined factor to adjust the pitch. 
            If non-zero, this factor will be used instead of computing it from the reference audio. Defaults to 0.
        speech_enroll (bool, optional): Flag indicating whether the pitch adjustment is for speech enrollment. Defaults to False.

    Returns:
        Tuple[numpy.ndarray, float]: Tuple containing the adjusted pitch information (source_f0) and the pitch shift factor (factor).

    """
    source_f0 = compute_pitch(wav_path)
    nonzero_indices = np.nonzero(source_f0)
    source_mean = np.mean(source_f0[nonzero_indices], axis=0)

    if predefined_factor != 0.:
        print(f'Using predefined factor {predefined_factor}.')
        factor = predefined_factor
    else:
        # Compute mean and std for pitch with the reference audio
        ref_wav, ref_fs = load_wav(ref_path)
        ref_f0 = ParselMouth_F0(ref_wav, ref_fs)
        nonzero_indices = np.nonzero(ref_f0)
        ref_mean = np.mean(ref_f0[nonzero_indices], axis=0)
        factor = ref_mean / source_mean
        if speech_enroll:
            factor = factor * 1.2
        print(f'pitch shift factor: {factor:.2f}')

    # Modify f0 to fit with different persons
    source_f0 = source_f0 * factor

    return source_f0, factor


def go(files, audio_dir, pitch_dir, rank):
    if rank == 0:
        pb = tqdm(files)
    else:
        pb = files

    for file in pb:
        f0 = compute_pitch(file, (pitch_dir/file.relative_to(audio_dir)).with_suffix('.npy'))


def main(args):
    data_root = Path(args.data_root)
    pitch_dir = Path(args.pitch_dir) if args.pitch_dir is not None else data_root/'pitch'
    n_p = args.n_cpu
    files = list(data_root.rglob('*.wav'))
    print(f"{len(files)} files to extract")
    ps = []
    for i in range(n_p):
        p = Process(
            target=go,
            args=(files[i::n_p], data_root, pitch_dir, i)
        )
        ps.append(p)
        p.start()
    for i in range(n_p):
        ps[i].join()


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from pathlib import Path
    from multiprocessing import Process

    parser = argparse.ArgumentParser(description="Compute pitch and loudness")

    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--pitch_dir', type=str)
    parser.add_argument('--n_cpu', type=int, default=1)

    args = parser.parse_args()
    main(args)