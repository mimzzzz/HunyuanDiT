from index_kits import IndexV2Builder
import glob
import os

# Your arrow directory
arrow_folder = "datasets/arrow_index1/" 
arrow_files = glob.glob(os.path.join(arrow_folder, "*.arrow"))

builder = IndexV2Builder(arrow_files)
builder.save("datasets/arrow_index1/train_dataset.json")
