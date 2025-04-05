# This file is to combine for each sequence athe train and test camera metadata such that 
# we have for each sequence only one metadata file

import json
import os
from pathlib import Path

SEQUENCES = ['basketball']
BASE_DATASET_DIR = "datasets/panoptic"
OUTPUT_DIR = Path("datasets/panoptic_torch")

for seq in SEQUENCES:
    test_meta_path = json.load(open(os.path.join(BASE_DATASET_DIR, f"{seq}/test_meta.json"), 'r')) 
    train_meta = json.load(open(os.path.join(BASE_DATASET_DIR, f"{seq}/train_meta.json"), 'r')) 
    
    combined_metadata = {}
    # Static generic propreties of each file are w and h
    # Then, frame specific informations
    
    
    
    
    # Save the combined metadata for the specific sequence