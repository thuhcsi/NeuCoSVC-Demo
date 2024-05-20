# Neural Concatenative Singing Voice Conversion v2

NeuCoSVC: [[Paper](https://arxiv.org/abs/2312.04919)] &emsp; [[Demo Page](https://thuhcsi.github.io/NeuCoSVC/)] &emsp; [[Checkpoints (google dirve)](https://drive.google.com/file/d/1QjoQ6mt7-OZPHF4X20TXbikYdg8NlepR/view?usp=drive_link)] <br>

NeuCoSVC2: [[gradio (EN) (coming soon)](https://openxlab.org.cn/apps/detail/Kevin676/NeuCoSVC2)] &emsp; [[gradio (中文)](https://openxlab.org.cn/apps/detail/Kevin676/NeuCoSVC2)] &emsp; [[Checkpoints (google dirve)](https://drive.google.com/file/d/1yDnT4Ah8Nlzq3QIff4ur4rz5CVpwYoip/view?usp=drive_link)]<br>
[[Video demo (BiliBili) from Kevin](https://www.bilibili.com/video/BV1fz42127wX/?spm_id_from=333.337.search-card.all.click)] <be>

This repository contains the official implementation of NeuCoSVC2, which is an enhanced version of [NeuCoSVC](https://arxiv.org/abs/2312.04919). The model has been trained on an extensive internal dataset comprising approximately 500 hours of singing voice data, supplemented by various open-source speech datasets. With the integration of the Phoneme Hallucinator, we have achieved significant improvements in audio quality, naturalness, and voice similarity. These enhancements are particularly noticeable when using shorter segments of reference audio.

## 📝 Changelog
- __[2024.05.20]__:  Release the training code to the repository and updated the checkpoint. Please use the [provided link](https://drive.google.com/file/d/1yDnT4Ah8Nlzq3QIff4ur4rz5CVpwYoip/view?usp=drive_link) to download the `G_150k.pt` file again for inference.

- __[2024.05.16]__: Release the inference code and model checkpoint for NeuCoSVC2.

## 🔧 Setup Instructions

### Environment Configuration

For an optimal development environment, we suggest using Anaconda to manage the project's dependencies. The provided `requirements.txt` outlines all necessary packages (including Torch 2.0.1 and TensorFlow 2.15). Please note that TensorFlow 2.15 requires CUDA 12.2 or above for GPU inference. To create and activate this environment, execute the following commands:

```bash
conda create -n NeuCoSVC2 python=3.10
conda activate NeuCoSVC2
pip install -r requirements.txt
```

Additionally, the [REAPER](https://github.com/google/REAPER) tool is required for pitch extraction. Please download and compile *reaper*. Afterwards, ensure to update the path to the reaper executable in line 15 of the script located at [utils/pitch_extraction.py](utils/pitch_extraction.py).

### Checkpoints

To set up the checkpoints for the project, you'll need to acquire the pre-trained models for the WavLM Large Encoder, the NeuCoSVC model and the Hallucinator model.

1. **WavLM Large Encoder:**
   - Visit the [WavLM repository](https://github.com/microsoft/unilm/tree/master/wavlm) hosted by Microsoft on GitHub.
   - Follow the instructions provided there to download the `WavLM-Large.pt` checkpoint file, then put the `WavLM-Large.pt` file in the `pretrained` folder.

2. **NeuCoSVC Model:**
   - Access the provided [Google Drive link](https://drive.google.com/file/d/1yDnT4Ah8Nlzq3QIff4ur4rz5CVpwYoip/view?usp=drive_link) to download the model.
   - Put the `G_150k.pt` file in the `pretrained` folder.

3. **Hallucinator Model:**
   - Access the provided [Dropbox link](https://www.dropbox.com/scl/fi/ytj3mwkf1fd0no4jtg7r7/weights.zip?rlkey=ilyxue0gpuppyzn6u01bbjiy9&dl=1) to download the model. After extracting the compressed file, place the `weights` folder into the `modules/Phoneme_Hallucinator/exp/speech_XXL_cond/` directory.

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
python infer.py --src_wav_path source-wav-path --ref_wav_path reference-wav-path --out_dir out-directory --speech_enroll
```

## 🏋️ Model Training

### Data Preparation

To prepare your data for training, you need to follow a series of preprocessing steps. These steps involve extracting pitch features and pre-matching features for each audio piece, and finally splitting the dataset and generating metadata files. Here's how you can perform each step, taking the OpenSinger dataset as an example. Please note that audio files need to be **resampled to 24kHz**.

### 1. Extract Pitch

Run the following command to extract pitch and loudness features from your audio dataset:

```bash
python -m utils.pitch_extraction --data_root dataset-root --pitch_dir dir-for-pitch --n_cpu 8
```

- `--data_root` specifies the root directory of your dataset.
- `--pitch_dir` is the directories where you want to save the pitch features. If not specified, they will be saved in the `pitch` folder under `dataset-root`.
- `--n_cpu` indicates the number of CPU cores to use for processing.

### 2. Extract Pre-Matching Features

To extract pre-matching features, use the following command:

```bash
python -m dataset.prematch_dataset --data_root dataset-root --out_dir dir-for-wavlm-feats
```

- This script will process the audio files and extract the 6th layer features of the WavLM model. 
- `--out_dir` is the directory where you want to save the WavLM features. If not specified, they will be saved in the `wavlm_features` folder under `dataset-root`.

### 3. Split Dataset and Generate Metadata

Finally, to split the dataset and generate metadata files, run:

```bash
python dataset/metadata.py --data_root dataset-root
```

- This will create three CSV files in the `dataset` folder. These files correspond to the data path of the training set, validation set, and test set, respectively. These three files will be used in subsequent model training. 
- By default, singing audio clips from the 26th and 27th male singers(OpenSinger/ManRaw/26(7)\_\*/\*.wav) and 46th and 47th female singers(OpenSinger/WomanRaw/46(7)\_\*/\*.wav) are considered as the test set. The remaining singers' audio files are randomly divided into the train set and the valid set in a 9:1 ratio.
- If you need to specify where to read the features from, you can use `--wavlm_dir` and `--pitch_dir` to point to the respective directories. If not specified, it will look for features in the `wavlm_features` and `pitch` folders under `data_root`.

### Decoder Training

```bash
python train.py --input_training_file dataset/opensinger-train.csv --input_validation_file dataset/opensinger-valid.csv --ckpt_dir pretrained --config configs/config.json
```

The two CSV files mentioned in the command line, `opensinger-train.csv` and `opensinger-valid.csv`, are obtained during the previous step of data preparation. To modify the training configurations or model parameters, you can edit the `configs/config.json` file. 
Additionally, you can specify the training epochs, stdout interval, checkpoint interval, summary interval, and validation interval by using the following options: `--training_epochs`, `--stdout_interval`, `--checkpoint_interval`, `--summary_interval`, and `--validation_interval`, respectively. 

## Acknowledgements

This work is inspired by [kNN-VC](https://github.com/bshall/knn-vc/tree/master) and built on the [U-net SVC](https://www.isca-speech.org/archive/interspeech_2022/li22da_interspeech.html) frameworks. 

We have incorporated publicly available code from the [kNN-VC](https://github.com/bshall/knn-vc/tree/master), [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) and [Phoneme_Hallucinator](https://github.com/PhonemeHallucinator/Phoneme_Hallucinator)projects. We would like to express our gratitude to the authors of kNN-VC, WavLM and Phoneme_Hallucinator for sharing their codebases. Their contributions have been instrumental in the development of our project.

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

[![Star History Chart](https://api.star-history.com/svg?repos=thuhcsi/NeuCoSVC&type=Date)](https://star-history.com/#thuhcsi/NeuCoSVC&Date)
