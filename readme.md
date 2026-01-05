# pix2pixhd-medical

Research code for conditional mask-to-image medical image synthesis using pix2pixHD.

This repository accompanies an arXiv research report and evaluates pix2pixHD-based synthesis across:
- LGG brain MRI tumour data
- ISIC skin lesion data

The focus is on cross-domain behaviour, training stability, and downstream segmentation utility.

## Reproducibility note
The code is provided for transparency and experimentation. Some preprocessing steps were simplified or omitted in the public implementation for clarity. These changes do not affect the qualitative conclusions reported in the paper.

## Data
Datasets are not included. Place datasets under a `data/` directory following the expected structure.
