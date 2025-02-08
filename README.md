# KidneyVision
Pipeline to generate data, train/test, and run inference on synthetic data created from crops of kidney stones on backgrounds. 

## Folder Structure
    * ```configs``` for json files to specify data generation and training configurations. See ```configs/schema.txt``` for example config file for ```scripts/datagen.py```
    * ```data``` stores the background images, crop images, and videos (for inference)
    * ```models``` stores .pt files used for training (e.g. yolov8)
    * ```scripts``` stores datageneration script ```datagen.py``` and training script ```train_test.py```

## Example usage

### Dataset Generation
```sh
python ./scripts/datagen.py \
--config_dir ./configs/2025-2-8_01.json" \
--train_size 1000 \
--val_size 200 \
```

### Training
```sh
python ./scripts/train_test.py \
--name 2025-2-8_01 \
--batch 256 \
--epochs 400 \
```