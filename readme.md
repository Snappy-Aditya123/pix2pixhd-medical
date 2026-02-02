# pix2pixhd-medical

This repository contains research code for conditional mask-to-image medical image synthesis based on the pix2pixHD framework.

The project evaluates pix2pixHD-based synthesis across two distinct medical imaging domains:
- LGG brain tumour MRI
- ISIC skin lesion dermoscopic images

The primary goal is to study cross-domain behaviour, training stability, and the impact of synthetic data on downstream segmentation performance under limited-data conditions.

---

## Method Overview
The image synthesis model is based on the pix2pixHD architecture originally released by NVIDIA. The implementation in this repository adapts the core mask-conditioned image-to-image translation framework for medical imaging tasks, with dataset-specific preprocessing and conditioning strategies.

Downstream evaluation is performed using a U-Net segmentation model trained with real data only and with a combination of real and synthetic images.

---

## Code Attribution
This project builds upon the pix2pixHD framework originally developed by NVIDIA and released under a BSD-style license.

- Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, et al.  
  *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* (CVPR 2018)  
  Official implementation: https://github.com/NVIDIA/pix2pixHD

Portions of the pix2pixHD codebase have been adapted and modified to support medical image synthesis, dataset-specific preprocessing, and downstream segmentation evaluation. The original copyright notice and license are retained in accordance with the BSD license.


---

## Datasets
This repository does **not** include any datasets.

The following public datasets were used in the accompanying research report:

- **LGG Brain MRI Dataset**  
  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

- **ISIC Skin Lesion Dataset (2018 Challenge)**  
  https://challenge.isic-archive.com/data/#2018

Users are responsible for downloading the datasets and placing them under a local `data/` directory following the expected structure.

---

## Preprocessing
Preprocessing for the skin lesion dataset follows the methodology described in:

Bissoto, A., Perez, F., Valle, E., Avila, S.  
*Skin Lesion Synthesis with Generative Adversarial Networks*,  
OR 2.0 / ISIC Workshop @ MICCAI 2018.




---

## Reproducibility Note
This code is provided for transparency and research reference. Some preprocessing steps were simplified or omitted in the public implementation for clarity and usability. These changes do not affect the qualitative conclusions reported in the accompanying research paper, but exact numerical reproduction is not guaranteed.

---

## License
This project is released under the MIT License.

