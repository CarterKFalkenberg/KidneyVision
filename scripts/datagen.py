import cv2
import numpy as np
import os
import sys
import pickle
import random
import imutils
import argparse
import shutil
import math
from PIL import Image
from pathlib import Path
import json
from typing import List, Tuple

def resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Resizes an image by some factor
    Args:
        image: GBR or GBRA image, np.ndarray shape H, W, 3 or H, W, 4
    Returns:
        resized image: image scaled by factor
    """
    new_height = int(image.shape[0] * factor)
    new_width = int(image.shape[1] * factor)

    if image.shape[-1] == 4:  # GBRA
        # Split the image into RGB and alpha channels and resize separately
        image_gbr = image[:, :, :3]
        resized_gbr = cv2.resize(image_gbr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        image_alpha = image[:, :, 3]
        resized_alpha = cv2.resize(image_alpha, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        result = np.dstack((resized_gbr, resized_alpha))
    else:  # GBR
        result = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return result

def load_images(images_dir: str, include_alpha: bool = True) -> List[np.ndarray]:
    """Loads and returns images from images_dir.
    
    Args:
        images_dir: Path to directory containing images
        include_alpha: Whether or not to include alpha channel when loading 
    Returns: 
        List of np arrays, each representing a crop in BGR or BGRA format
    """
    READ_MODE = cv2.IMREAD_UNCHANGED if include_alpha else cv2.IMREAD_COLOR_BGR
    images = []
    for filename in os.listdir(images_dir):
        image = cv2.imread(filename, READ_MODE)
        images.append(image)
    return images

def duplicate_check(dataset_path: str):
    """Ensures that dataset does not already exist with identical name. 
    If one does, ask user to confirm its deletion or exit the program.
    
    Args:
        dataset_path: directory to save generated dataset 
    """
    dataset_path = os.path.join(output_dir, config["dataset_name"])
    if os.path.exists(dataset_path):
        print("[WARNING] Dataset with identical name already exists")
        response = input("Would you like to overwrite the existing dataset? [Y/n]: ")
        if response == 'Y':
            shutil.rmtree(dataset_path)
            print("[CONFIRMATION] Existing dataset has been deleted and overwritten")
        else:
            print("[CONFIRMATION] Existing dataset will not be deleted. Exiting program...")
            sys.exit()
            
def generate_datset_dir(dataset_path: str):
    """Generate dataset directory tree:
    output_dir/ 
        dataset_name/
            images/
                train/
                val/
            labels/
                train/
                val/
    
    Args:
        dataset_path: directory to save generated dataset 
    """
    os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, "images"))
    os.makedirs(os.path.join(dataset_path, "labels"))
    for set_name in ["train", "val"]:
        os.makedirs(os.path.join(dataset_path, "images", set_name))
        os.makedirs(os.path.join(dataset_path, "labels", set_name))
        
def write_labels(label: List[int], filepath: str):
    """Writes crops labels to filepath
    
    Args:
        labels: label for the crop on the background
        filepath: txt file to write the labels to 
    """
    with open(filepath, "w") as f:
        if not label:    # creates empty file 
            break
        line = str(labels[i][0])
        for j in range(1, len(labels[i])):
            line += " " + str(labels[i][j])
        f.write(line+"\n")
        
def get_segmentation_annotation(crop, scaling_factor: int = 0.001) -> List[Tuple[int, int]]: # Segmentation annotation test
    """Gets the annotation for the segmentation of the given crop
    1. Sets all gbr values to 0 if alpha == 0
    2. Converts image to grayscale
    3. Get binary mask (1 if in crop, 0 if not in crop for each pixel)
    4. Find all contours and choose largest (largest is entire crop)
    5. Estimate contour with fewer points
    Args:
        crop: crop image, shape H, W, 4
        scaling_factor: adjusts the number of points for contour estimation (lower => more points)
    Returns:
        points: list of all points in the segmentation of the crop
    """
    
    crop_bgr = crop[:, :, :3].copy()
    # set all values to 0 if alpha channel is 0
    crop_bgr[crop[:, :, 3] == 0] = 0 
    
    gray_crop = cv2.cvtColo(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = scaling_factor * cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    points = approximated_contour.reshape(-1, 2).tolist()
    
    return points

def normalize_annotation(annotation: List[Tuple[int, int]], height: int, width: int, h_offset: int, w_offset: int) -> List[Tuple[int, int]]: # Segmentation annotation test
    """Normalizes each (x, y) coordinate in the annotation list according to the size of the background
    
    Args:
        annotation: Annotations generated from 'get_segmentation_annotation'
        height: height of background image
        width: width of background image
        h_offset: height offset of crop on background image
        w_offset: width offset of crop on background image
    """
    h_offset = h_offset / height
    w_offest = w_offset / width

    # Normalize each point in the annotation list
    normalized_annotation = [(w_offset + (w / width), h_offset + (h / height)) for w, h in annotation]

    return normalized_annotation
            
def generate_augmented_image(
    background: np.ndarray, 
    crop: np.ndarray, 
    resize_min: float, 
    resize_max: float,
    random_pos: bool, 
    random_rotate: bool, 
    blur_res: bool
) -> Tuple(np.ndarray, List[float]):
    """Pastes crop onto background according to config parameters
    Args:
        background: background image, shape H, W, 3
        crop: crop image, shape H', W', 4 
            * Assumes crop can fit into background
        resize_min: between (0,1]. Minimum resize % 
        resize_max: > 0. Maximum resize % 
        random_pos: sets random x, y position for crop if true
        random_rotate: rotates crop by rando theta if true
        blur_res: applies gaussian blur to resulting image if true 
    Returns:
        background: background image with crop
        annotations: 1d list of crop segmentation annotations
    """
    background, crop = background.copy(), crop.copy()
    
    if resize_min != 1 or resize_max != 1:
        factor = np.random.uniform(resize_min, resize_max)
        crop = resize(crop, factor)    
    
    if random_rotate:
        crop = imutils.rotate(crop, np.random.randint(0, 360))
    
    max_w = background.shape[1] - crop.shape[1]
    max_h = background.shape[0] - crop.shape[0]
    
    h, w =  (np.random.randint(0, max_h), np.random.randint(0, max_w)) if random_pos else (0, 0)
    
    # get region of interest of background
    roi = background[h: h + crop.shape[0], w: w + crop.shape[1]]
    
    crop_bgr = crop[:, :, :3]
    crop_alpha_norm = crop[:, :, 3] / 255.0
    
    # blend crop and background
    for channel in range(3): # each color channel
        roi[:, :, c] = (alpha * crop_bgr[:, :, c]) + ((1 - alpha) * roi[:, :, c])

    background[h: h + crop.shape[0], w: w + crop.shape[1]] = roi
    if blur_res:
        background = cv2.GaussianBlur(background, (13, 13), 0)
    
    annotation = get_segmentation_annotation(crop)
    annotation_norm = normalize_annotaiton(annotation, *background.shape, h, w)
    
    # flatten and add class label '0' to the annotation (0 => stone)
    flattened_annotation = [0] + annotation_norm.flatten() 
    return (background, flattened_annotation)
    
    

def run(args):
    with open(args.config) as f:
        config = json.load(f)
    DATASET_PATH = os.path.join(output_dir, config["dataset_name"])    
    RESIZE_MIN, RESIZE_MAX = config.get("resize_min", 1), config.get("resize_max", 1)
    RANDOM_POS = config["random_pos"]
    RANDOM_ROTATE = config["random_rotate"]
    BLUR_RES = config["blur_res"]
        
    backgrounds = load_images(args.backgrounds_dir)
    crops = load_images(args.crop_dir, include_alpha = False)
    
    # make sure not overwriting existing dataset 
    duplicate_check(dataset_path)
    
    generate_dataset_dir(DATASET_PATH)
    
    subsetSize = {
        "train": args.trainSize,
        "val": args.valSize,           
    }
    
    for set_name, size in subsetSize.items():
        image_path = os.path.join(DATASET_PATH, "images", set_name)
        labels_path = os.path.join(DATASET_PATH, "labels", set_name)
        for i in range(size):
            # add random crop to random background  
            background = np.random.randint(0, len(backgrounds))
            crop = np.random.randint(0, len(crops))
            
            image, label = generate_augmented_image(
                background, 
                crop,
                RESIZE_MIN, RESIZE_MAX,
                RANDOM_POS,
                RANDOM_ROTATE,
                BLUR_RES    
            )
            cv2.imwrite(os.path.join(image_path, f"image_{i}"), image)
            write_labels(label, os.path.join(labels_path, f"image_{i}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Kidney Stone Dataset")
    
    # dataset size
    parser.add_argument('--trainSize', type=int, default=2000, help='training set size, default 2000')
    parser.add_argument('--valSize', type=int, default=400, help='validation set size, default 400')
    
    # override defaults
    parser.add_argument("--config", type=str, default="./configs", help="Config file")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Directory to save generated dataset dir")
    parser.add_argument("--backgrounds_dir", type=str, default="./data/backgrounds", help="Directory containing background images")
    parser.add_argument("--crops_dir", type=str, default="./data/crops", help="Directory containing cropped stone images")

    args = parser.parse_args()
    run(args)