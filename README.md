# KidneyVision
Pipeline to generate data and train/test on synthetic data created from crops of kidney stones on backgrounds using openCV + yolov8. 


## Folder Structure
 - ```configs``` for json files to specify data generation and training configurations. See ```configs/schema.txt``` for example config file for ```scripts/datagen.py```
 - ```data``` stores the background images, crop images, and videos (for inference)
 - ```models``` stores .pt files used for training (e.g. yolov8)
 - ```scripts``` stores datageneration script ```datagen.py``` and training script ```train_test.py```

## Example usage

### Dataset Generation
```sh
python ./scripts/datagen.py \
--config_dir ./configs/2025-2-8_01.json \
--train_size 1000 \
--val_size 200 \
```
Dataset generation script randomly chooses a crop and a background and combines them into one segmented image. Different image augmentation
techniques can be applied such as resizing and rotating the crop, random positioning of the crop, and gaussian blurring. 

### Training
```sh
python ./scripts/train_test.py \
--name 2025-2-8_01 \
--batch 256 \
--epochs 400 \
```
Training runs on YOLOv8 models and yields high validation accuracy. Accuracy on inference videos is decent but currently being improved.

### Features in development for inference:
- "Optical flow" and/or past frame information to reduce incorrect segmentations
- 3D kidney reconstruction in ISAAC-SIM to generate more data
- Crop edge blurring (alpha-blur) to reduce harshness of crop pasted onto background
- Random noise additions to simulate the flakiness of the stone as it is being lasered 
