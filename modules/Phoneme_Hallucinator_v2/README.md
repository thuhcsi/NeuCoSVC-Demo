# Phoneme_Hallucinator
This is the repository of the paper "Phoneme Hallucinator: One-shot Voice Conversion via Set Expansion" accepted by AAAI-2024. Some audio samples are provided [here](https://phonemehallucinator.github.io/).

## Inference Tutorial
1. If you only want to run our VC pipeline, please download `Phoneme Hallucinator DEMO.ipynb` in this repo and run it in google colab.
   
## Training Tutorial
1. Prepare environment. Require `Python 3.6.3` and the following packages
   ```
   pillow == 8.0.1
   torch == 1.10.2
   tensorflow == 1.15.5
   tensorflow-probability == 0.7.0
   tensorpack == 0.9.8
   h5py == 2.10.0
   numpy == 1.19.5
   pathlib == 1.0.1
   tqdm == 4.64.1
   easydict == 1.10
   matplotlib == 3.3.4
   scikit-learn == 0.24.2
   scipy == 1.5.4
   seaborn == 0.11.2
   ```
3. To prepare the training set, we need to use WavLM to extract speech representations. Go to [kNN-VC repo](https://github.com/bshall/knn-vc) and follow its instructions to extract speech representations. Namely, after placing LibriSpeech dataset in a correct location, run the command:

   `python prematch_dataset.py --librispeech_path /path/to/librispeech/root --out_path /path/where/you/want/outputs/to/go --topk 4 --matching_layer 6 --synthesis_layer 6`

   Note that we don't use the "--prematch" option, becuase we only need to extract representations, not to extract and then perform kNN regression.

4. After the above step, you can get a `--out_path` folder with three subfolders `train-clean-100`, `test-clean` and `dev-clean` where each folder contains the speech representation files (".pt").
5. Go to our repo `./dataset/speech.py` and change the variables `path_to_wavlm_feat` and `tfrecord_path` accordingly. You need to change `path_to_wavlm_feat` to where the speech representations are stored in the previous step.
6. Start Training by the following command: 
   `python scripts/run.py --cfg_file=./exp/speech_XXL_cond/params.json --mode=train`
   
   If `tfrecord_path` doesn't exist, our codes will create tfrecords and save them to `tfrecord_path` before training starts. Note that if you encounter numerical issues ("NaN, INF") when the training starts, just try re-run the command multiple times. Training los will be saved to `./exp/speech_XXL_cond/`.
