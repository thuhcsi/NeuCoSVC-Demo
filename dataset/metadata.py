import os
import csv
import random
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path


def GetMetaInfo(wav_path):
    relative_path = wav_path.relative_to(data_root)
    wavlm_path = (wavlm_dir/relative_path).with_suffix('.pt')
    pitch_path = (pitch_dir/relative_path).with_suffix('.npy')
    assert os.path.isfile(wavlm_path), f'{wavlm_path} does not exist.'
    assert os.path.isfile(pitch_path), f'{pitch_path} does not exist.'

    return [str(wav_path), str(wavlm_path), str(pitch_path)]


def SplitDataset(wav_list:list[Path], train_valid_ratio=0.9, test_spk_list=['M26','M27','W46','W47']):
    '''
    Split the dataset into train set, valid set, and test set. 
    By default, it considers the OpenSinger dataset's 26th and 27th male singers (M26, M27) and 
    46th and 47th female singers (W46, W47) as the test set. 
    The remaining singers' audio files are randomly divided into the train set and the valid set in a 9:1 ratio.

    Args:
        wav_list (list[Path]): List of Path objects representing the paths to the wav files.
        train_valid_ratio (float, optional): Ratio of the dataset to be used for training and validation. Defaults to 0.9.
        test_spk_list (list[str], optional): List of speaker IDs to be included in the test set. Defaults to ['M26', 'M27', 'W46', 'W47'].

    Returns:
        Tuple[list[Path], list[Path], list[Path]]: Tuple containing the train set, valid set, and test set as lists of Path objects.

    '''
    train_list = []
    valid_list = []
    test_list = []

    for wav_file in wav_list:
        singer = wav_file.parent.parent.name[0] + wav_file.stem.split('_')[0]
        if singer not in test_spk_list:
            train_list.append(wav_file)
        else:
            test_list.append(wav_file)

    random.shuffle(train_list)

    train_valid_split = int(len(train_list) * train_valid_ratio)

    train_list, valid_list = train_list[:train_valid_split], train_list[train_valid_split:]

    return train_list, valid_list, test_list


def GenMetadata(wav_list, mode):
    '''
    generate the metadata file for the dataset
    '''
    results = Parallel(n_jobs=10)(delayed(GetMetaInfo)(wav_path) for wav_path in tqdm(wav_list))
    
    with open(f"dataset/opensinger-{mode}.csv", 'w') as f:
        writer = csv.writer(f, delimiter='\t',)
        writer.writerow(["audio_path", "feat_path", "pitch_path"])
        writer.writerows(results)
    
    return


def main(args):
    global data_root, wavlm_dir, pitch_dir, ld_dir
    data_root = Path(args.data_root)
    wavlm_dir = Path(args.wavlm_dir) if args.wavlm_dir is not None else data_root/'wavlm_features'
    pitch_dir = Path(args.pitch_dir) if args.pitch_dir is not None else data_root/'pitch'
    wav_list = list(data_root.rglob('*.wav'))
    train_list, valid_list, test_list = SplitDataset(wav_list)

    GenMetadata(train_list, 'train')
    GenMetadata(valid_list, 'valid')
    GenMetadata(test_list, 'test')
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        required=True, type=str, help='Directory of audios for the dataset.'
    )
    parser.add_argument(
        '--wavlm_dir',
        type=str, help='Directory of wavlm features for the dataset.'
    )
    parser.add_argument(
        '--pitch_dir',
        type=str, help='Directory of pitch for the dataset.'
    )

    args = parser.parse_args()
    main(args)