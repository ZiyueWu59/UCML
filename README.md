# UCML
Official code for ACM MM'23 paper: "Weakly-supervised Video Scene Graph Generation via Unbiased Cross-modal Learning".

# Data preparing
## Download our prepared data and model by [this link](https://drive.google.com/drive/folders/1ZZsHz50Q2wSQnLulkFHLwn42xvHDscYt?usp=drive_link), and put them in the following folder as
```
├── datasets
│   ├── cache                      
│   │   ├── PKU_v1_VidVRDtest_th_5-50-100.pkl
│   ├── vidvrd-dataset
│   │   ├── train
│   │   ├── test
│   │   └── videos
│   └── GT_json_for_eval
│   │   └── VidVRDtest_gts.json
│   ├──preprocess_data                # prerpocess trajectory features
│   ├── tracking
│   └── videovrd_detect_tracking              
│   └── ...
├── ...
```


# Evaluation
```
python tools/eval_vidvrd.py --cfg_path experiments/config_.py --ckpt_path vidvrd_best.pth --gt_relations_path datasets/GT_json_for_eval/VidVRDtest_gts.json --save_tag vidvrd
```


# Citation
If our work is helpful for your research, please cite our publication:
```
@inproceedings{wu2023weakly,
  title={Weakly-supervised Video Scene Graph Generation via Unbiased Cross-modal Learning},
  author={Wu, Ziyue and Gao, Junyu and Xu, Changsheng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4574--4583},
  year={2023}
}
```

# Acknowledgement
This repo contains modified codes from [VidSGG-BIG](https://github.com/Dawn-LX/VidSGG-BIG).
We sincerely thank the owners of the great repo!
