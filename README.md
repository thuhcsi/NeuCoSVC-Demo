# Neural Concatenative Singing Voice Conversion

NeuCoSVC: [[Paper](https://arxiv.org/abs/2312.04919)] &emsp; [[Demo Page](https://thuhcsi.github.io/NeuCoSVC/)] &emsp; [[Checkpoints (google drive)](https://drive.google.com/file/d/1QjoQ6mt7-OZPHF4X20TXbikYdg8NlepR/view?usp=drive_link)] <br>

NeuCoSVC2: [[Gradio (English) (coming soon)](https://openxlab.org.cn/apps/detail/Kevin676/NeuCoSVC2)] &emsp; [[Gradio (中文)](https://openxlab.org.cn/apps/detail/Kevin676/NeuCoSVC2)] &emsp; [[Video Demo (BiliBili) from Kevin](https://www.bilibili.com/video/BV1fz42127wX/?spm_id_from=333.337.search-card.all.click)] <br>

This repository contains the official implementation of NeuCoSVC, a versatile model for any-to-any singing voice conversion. Please note that NeuCoSVC has been upgraded to NeuCoSVC2. For the latest version, switch to the NeuCoSVC2 branch!

![NeuCoSVC](./img/Architecture.png)

Figure: The structure of the proposed SVC system: (a) the SSL feature extracting and matching module; (b) the neural harmonic signal generator; (c) the audio synthesizer.

## 🚩 New Features/Updates
- ✅ May 16, 2024: The release of NeuCoSVC2 marks a significant update from its predecessor, NeuCoSVC. Enhancements include training on an expanded dataset and the integration of the Phoneme Hallucinator. Access the latest version on the NeuCoSVC2 branch!
- ✅ Dec. 28, 2023: We proudly introduced NeuCoSVC, our initial singing voice conversion model.

## 🔧 Setup Instructions

### Environment Configuration

For an optimal development environment, we suggest using Anaconda to manage the project's dependencies. The provided `requirements.txt` outlines all necessary packages (including Torch 2.0.1 with cu117 support). To create and activate this environment, execute the following commands:

```bash
conda create -n NeuCoSVC python=3.10.6
conda activate NeuCoSVC
pip install -r requirements.txt
```

For those interested in the exact environment used during development, refer to the comprehensive list in `requirements_all.txt`.

Additionally, the [REAPER](https://github.com/google/REAPER) tool is required for pitch extraction. Please download and compile *reaper*. Afterwards, ensure to update the path to the reaper executable in line 60 of the script located at [utils/pitch_ld_extraction.py](utils/pitch_ld_extraction.py).

### Checkpoints

To set up the checkpoints for the project, you'll need to acquire the pre-trained models for both the WavLM Large Encoder and the FastSVC model with neural harmonic filters.

1. **WavLM Large Encoder:**
   - Visit the [WavLM repository](https://github.com/microsoft/unilm/tree/master/wavlm) hosted by Microsoft on GitHub.
   - Follow the instructions provided there to download the `WavLM-Large.pt` checkpoint file.

2. **FastSVC Model with Neural Harmonic Filters:**
   - Access the provided [Google Drive link](https://drive.google.com/file/d/1QjoQ6mt7-OZPHF4X20TXbikYdg8NlepR/view?usp=drive_link) to download the trained FastSVC model.

3. **Organizing the Checkpoints:**
   - Once you have both files, you'll need to place them in the correct directory within your project.
   - Create a folder named `pretrained` in the root directory of the project if it doesn't already exist.
   - Move the `WavLM-Large.pt` file and the `model.pkl` (assuming this is the correct name of the folder containing the FastSVC model) into the `pretrained` folder.

## 🌠 Inference

When you're ready to perform inference, it's important to ensure that your audio data meets the required specifications. Here's a checklist and a guide on how to proceed:

1. **Audio Requirements:**
   - The source waveform should have a sample rate of 24kHz.
   - Both the source and reference audio files should be in mono, not stereo.

2. **Using Speech as Reference:**
   - If you're using speech instead of singing as the reference audio, it's recommended to use the `--speech_enroll` flag. This will help the model better adapt the characteristics of the speech to singing.

3. **Pitch Shift:**
   - When using speech as the reference and aiming for a pitch shift, the pitch of the reference audio will be increased by a factor of 1.2. This adjustment helps bridge the natural pitch gap between spoken and sung vocals.

4. **Running the Inference Command:**
   - With your audio files prepared and placed in the correct directories, you'll run a command similar to the following:

```bash
python infer.py --src_wav_path src-wav-path --ref_wav_path ref-wav-path --out_path out-path --speech_enroll
```

## 🏋️ Model Training

### Data Preparation

Take the OpenSinger dataset as an example, the audio files need to be **resampled to 24kHz**. 

```
- OpenSinger_24k
    |- ManRaw/
    |   | - SingerID_SongName/
    |   |   | - SingerID_SongName_SongClipNumber.wav/
    |   |   | - ...
    |   | - ...
    |- WomanRaw/
    |   | - 0_光年之外/
    |   |   | - 0_光年之外_0.wav/ 
    |   |   | - ...
    |   | - ...
```

To prepare your data for training, you need to follow a series of preprocessing steps. These steps involve extracting pitch and loudness features, pre-matching features for each audio piece, and finally splitting the dataset and generating metadata files. Here's how you can perform each step:

### 1. Extract Pitch and Loudness

Run the following command to extract pitch and loudness features from your audio dataset:

```bash
python -m utils.pitch_ld_extraction --data_root dataset-root --pitch_dir dir-for-pitch --ld_dir dir-for-loudness --n_cpu 8
```

- `--data_root` specifies the root directory of your dataset.
- `--pitch_dir` and `--ld_dir` are the directories where you want to save the pitch and loudness features, respectively. If not specified, they will be saved in the `pitch` and `loudness` folders under `dataset-root`.
- `--n_cpu` indicates the number of CPU cores to use for processing.

### 2. Extract Pre-Matching Features

To extract pre-matching features, use the following command:

```bash
python -m dataset.prematch_dataset --data_root dataset-root --out_dir dir-for-wavlm-feats
```

- This script will process the audio files and extract features using the WavLM model. It uses the average of the last five layers' features from WavLM for distance calculation and kNN. It replaces and concatenates on the corresponding feature of the 6th layer in WavLM for audio synthesis. This configuration has shown improved performance in experiments.
- `--out_dir` is the directory where you want to save the WavLM features. If not specified, they will be saved in the `wavlm_features` folder under `dataset-root`.

### 3. Split Dataset and Generate Metadata

Finally, to split the dataset and generate metadata files, run:

```bash
python dataset/metadata.py --data_root dataset-root
```

- This will create train, validation, and test sets from your dataset. By default, singing audio clips from the 26th and 27th male singers(OpenSinger/ManRaw/26(7)\_\*/\*.wav) and 46th and 47th female singers(OpenSinger/WomanRaw/46(7)\_\*/\*.wav) are considered as the test set. The remaining singers' audio files are randomly divided into the train set and the valid set in a 9:1 ratio.
- If you need to specify where to read the features from, you can use `--wavlm_dir`, `--pitch_dir`, and `--ld_dir` to point to the respective directories. If not specified, it will look for features in the `wavlm_features`, `pitch`, and `loudness` folders under `data_root`.

### Decoder Training

```bash
# for single GPU training:
python start.py --data_root dataset-dir --config configs/config.json --cp_path pretrained
# for distributed multi GPUs training:
torchrun --nnodes=1 --nproc_per_node=4 start.py --data_root dataset-dir --config configs/config.json --cp_path pretrained
```

To modify the training configurations or model parameters, you can edit the `configs/config.json` file. 

## Acknowledgements

This work is inspired by [kNN-VC](https://github.com/bshall/knn-vc/tree/master) and built on the [U-net SVC](https://www.isca-speech.org/archive/interspeech_2022/li22da_interspeech.html) frameworks. 

We have incorporated publicly available codes from the [kNN-VC](https://github.com/bshall/knn-vc/tree/master) and [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) projects.

We would like to express our gratitude to the authors of kNN-VC and WavLM for sharing their codebases. Their contributions have been instrumental in the development of our project.

## Citation

If this repo is helpful with your research or projects, please kindly star our repo and cite our paper as follows:

```bibtex
@misc{sha2023neural,
      title={neural concatenative singing voice conversion: rethinking concatenation-based approach for one-shot singing voice conversion}, 
      author={Binzhu Sha and Xu Li and Zhiyong Wu and Ying Shan and Helen Meng},
      year={2023},
      eprint={2312.04919},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=thuhcsi/NeuCoSVC&type=Date)](https://star-history.com/#thuhcsi/NeuCoSVC&Date)

