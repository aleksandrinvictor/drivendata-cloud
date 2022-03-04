# 42nd place (top 5%) solution of [On Cloud N: Cloud Cover Detection Challenge](https://www.drivendata.org/competitions/83/cloud-cover/page/396/)
![](https://drivendata-public-assets.s3.amazonaws.com/cloud-cover-banner.jpg)

## Summary
The goal of this challenge was to detect cloud cover in satellite imagery to remove cloud interference. It is binary semantic segmentation task. Performance metric - Jaccard index.
My solution based on Unet model with ResNest50 backbone and pseudo labelling.

## Preprocess and augmentations
Train dataset provided by the organizers contained 11243 images. Many of them had wrong annotations (see fig below). So first of all I looked through all images in the dataset and removed 1633 images with wrong annotations.
![](./readme_imgs/bad_labels.png?raw=true "Examples of images with wrong labels")

Then dataset was splitted on folds with stratification by:
- photo location
- year
- month
- hour
- cloud coverage %

During training the following standard augmentation were used:
- Flips
- Normalization

Also I'v tried to apply different rotations and photometric distortions as well as more complex augmentations like:
- CutMix
- ChessMix
- Mosaic

But unfortunately it didn't improve my result.

## Model
My final model is Unet with ResNest50 backbone that was trained with lovasz loss and cosine annealing learning rate scheduler. I've used that model to make predictions for the images with wrong annotations and made pseudo labels with confidence threshold=0.9. That process was repeated 3 times to get more pseudo labels. Finally 1121 images out of 1633 was used as pseudo labels.

![](./readme_imgs/pseudo_labels.png?raw=true "Examples of pseudo-labels and original masks")

During inference vertical flip TTA was used. My local cv showed that 4-TTA works better but inference time was limited and I could use only one TTA. Also several other models was trained:
- FPN
- DeepLabv3+
- Linknet
- CloudNet

I'v tried to ensemble 3 best models (FPN-ResNet50, DeepLabv3-ResNeXt50, Unet-ResNest50) using trainable linear combination for every fold. That ensemble improved local cv but showed lower result for public lb.

# How to run
## Data structure
```
/data/
├── train_features/
    ├── adwp
        ├── B02.tif
        ├── B03.tif
        ├── B04.tif
        └── B08.tif
│   └── ...
├── train_labels/
    ├── adwp.tif
    └── ...
├── train.csv
/assets/ # checkpoints of trained models
/pretrained_models
├── resnest50.pth # pretrained on imagenet ResNest50 model (https://smp.readthedocs.io/en/latest/encoders.html#resnest)
```

## Environment
Two ways to setup environment:
1. conda environment described in `environment-gpu.yml`
2. Dockerfile

## Prepare data
```
python download_data.py --sas-url sas_westeurope.txt
python cloud/prepare_dataframe.py
python cloud/split_on_folds.py --data_path ./data/init_samples.csv --output_path ./data/init_folds
```

## Train models
### Stage 1
```
bash scripts train.sh configs/exp101_pslb_1.yml
python cloud/make_pseudo_labels.py --model_path ./assets/exp101_pslb_1 --train_data_path ./data/init_samples.csv --test_data_path ./data/train.csv --output_label_path ./data/pseudo_labels_1 --conf_thres 0.9
```
### Stage 2
```
bash scripts train.sh configs/exp101_pslb_2.yml
python cloud/make_pseudo_labels.py --model_path ./assets/exp101_pslb_2 --train_data_path ./data/pseudo_labels_1.csv --test_data_path ./data/train.csv --output_label_path ./data/pseudo_labels_2 --conf_thres 0.9
```
### Stage 3
```
bash scripts train.sh configs/exp101_pslb_3.yml
python cloud/make_pseudo_labels.py --model_path ./assets/exp101_pslb_3 --train_data_path ./data/pseudo_labels_2.csv --test_data_path ./data/train.csv --output_label_path ./data/pseudo_labels_3 --conf_thres 0.9
```
### Final model
```
bash scripts train.sh configs/exp102.yml
```
### Evaluation
```
python cloud/inference.py --model_path ./assets/exp102
```
