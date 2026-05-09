# Collaborative Transformers With Multi-Level Forensic Attention for Image Manipulation Localization(AAAI 2026)

Official repository for the AAAI 2026 paper **Collaborative Transformers With Multi-Level Forensic Attention for Image Manipulation Localization**.

[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/38250)

## Overview

**Co-Transformers** is a dual-Transformer framework for image manipulation localization. It collaboratively models both macroscopic semantic inconsistencies and microscopic forensic traces for accurate pixel-level localization of tampered regions.

The framework contains two complementary branches:

- **Hierarchical Edge-Supervised Transformer**, which captures boundary information and semantic cues from RGB images.
- **Cross-trace Extraction Transformer**, which focuses on high-frequency forensic noise artifacts.

We further introduce **Multi-Level Forensic Attention**, which enhances the model's ability to capture manipulation traces under different receptive fields and improves robustness against common image processing attacks.

<p align="center">
  <img src="images/model.svg" alt="Co-Transformers Framework Diagram" width="98%">
</p>

## Installation

Our training and testing pipeline is implemented based on [IMDL-BenCo](https://github.com/scu-zjz/IMDLBenCo). Please follow the dataset format and evaluation protocol provided by IMDL-BenCo.

Install IMDL-BenCo with:

```bash
pip install imdlbenco
```

For details about dataset preparation, benchmark configuration, and evaluation metrics, please refer to the official IMDL-BenCo documentation.

## Pretrained Weights and Checkpoints

We provide the pretrained weights used in our experiments, as well as the released Co-Transformers checkpoint, via [Google Drive](https://drive.google.com/drive/folders/1aL9zagvJjhwAVdZXf73EeJaxS74iCnc-?usp=sharing).

After downloading the weights, please update the checkpoint paths in the corresponding training or testing scripts.

## Training and Testing

Before running the scripts, please modify the dataset paths, checkpoint paths, and output directories according to your local environment.

### Training

```bash
bash train_cotransformer.sh
```

### Testing

```bash
bash test_cotransformer.sh
```

## Results

Due to an oversight, the checkpoint corresponding exactly to the results reported in the paper was lost. The currently released checkpoint was obtained by retraining the model under the same experimental setting.

The updated results are reported below.

**Bold** indicates the best result, and <u>underline</u> indicates the second-best result.

| Model | Coverage | Columbia | NIST16 | CASIAv1 | AutoSplice | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| MVSS-Net (Dong et al. 2022) | 0.5188 / 0.5503 | 0.7322 / 0.7772 | 0.2876 / 0.3377 | 0.5486 / 0.5707 | 0.3779 / 0.5570 | 0.4894 / 0.5586 |
| PSCC-Net (Liu et al. 2022) | 0.3791 / 0.4415 | 0.8640 / 0.8924 | 0.3681 / 0.4143 | 0.5472 / 0.5583 | **0.5531** / 0.6831 | 0.5423 / 0.5987 |
| IML-ViT (Ma et al. 2024a) | 0.5364 / 0.5852 | <u>0.9334 / 0.9721</u> | 0.1100 / 0.1503 | 0.7392 / 0.7512 | 0.2687 / 0.5676 | 0.5184 / 0.6053 |
| CAT-Net (Kwon et al. 2022) | 0.4272 / 0.5165 | 0.9151 / 0.9547 | 0.3787 / 0.3316 | 0.8140 / 0.8154 | 0.3870 / 0.6168 | 0.5844 / 0.6470 |
| TruFor (Guillaro et al. 2023) | 0.4573 / 0.5369 | 0.8845 / 0.9547 | 0.3480 / 0.4046 | 0.8176 / 0.8340 | 0.3830 / **0.6910** | 0.5781 / 0.6843 |
| Mesorch (Zhu et al. 2025) | <u>0.5862 / 0.6342</u> | 0.8903 / 0.9708 | <u>0.3921 / 0.4514</u> | <u>0.8398 / 0.8472</u> | 0.4004 / 0.6709 | <u>0.6138 / 0.7149</u> |
| **Co-Transformers (ours)** | **0.6543 / 0.6960** | **0.9561 / 0.9898** | **0.4551 / 0.4994** | **0.8548 / 0.8642** | <u>0.5360 / 0.6906</u> | **0.6913 / 0.7480** |

## Citation

If you find our work useful for your research or applications, please cite our paper:

```bibtex
@inproceedings{zhang2026collaborative,
  title={Collaborative Transformers with Multi-Level Forensic Attention for Image Manipulation Localization},
  author={Zhang, Jiwei and Feng, Wenbo and Wang, Siwei and Kou, Feifei and Yu, Haoyang and Niu, Shaozhang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={15},
  pages={12556--12563},
  year={2026},
  doi={10.1609/aaai.v40i15.38250}
}
```

## Acknowledgements

This project is implemented based on [IMDL-BenCo](https://github.com/scu-zjz/IMDLBenCo). We sincerely thank the authors and contributors of IMDL-BenCo for their excellent benchmark and codebase.

## Contact

If you have any questions, please feel free to open a GitHub issue. For direct inquiries, contact us at **[fwb@bupt.edu.cn](mailto:fwb@bupt.edu.cn)**.
