import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# WordNet IDs for selected classes (thank you chatgpt)
SELECTED_CLASSES = {
    'n01855672': 'goose',
    'n02324045': 'rabbit',
    'n02355227': 'squirrel',
    'n02802426': 'basketball',
    'n02978881': 'cassette',
    'n03179701': 'desk',
    'n03481172': 'hammer',
    'n03710637': 'T-shirt',
    'n03770679': 'minivan',
    'n03837869': 'oscilloscope',
    'n03938244': 'pillow',
    'n04009552': 'printer',
    'n04152593': 'screen',
    'n04347754': 'submarine',
    'n04376876': 'switch',
    'n04557648': 'water_bottle',
    'n04597913': 'wooden_spoon',
    'n07697537': 'hot_dog',
    'n07747607': 'orange',
    'n07753275': 'corn'
}

OUTPUT_DIR = './data' 
TARGET_SIZE = (80, 80)

api = KaggleApi()
api.authenticate()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for wnid, class_name in SELECTED_CLASSES.items():
    print(f"downloading {class_name} ({wnid})")
    api.dataset_download_file('dimensi0n/imagenet-256', file_name=f"{class_name}/000.jpg", path=OUTPUT_DIR)

    zip_path = os.path.join(OUTPUT_DIR, f"{wnid}.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(OUTPUT_DIR, f"{wnid}_{class_name}"))
    os.remove(zip_path)

print("done downloading")

# resize and clean because this dataset sucks
from PIL import Image

def resize_images(folder):
    for img_name in tqdm(os.listdir(folder), desc=f"resizing in {os.path.basename(folder)}"):
        img_path = os.path.join(folder, img_name)
        try:
            with Image.open(img_path) as img:
                if img.width < TARGET_SIZE[0] or img.height < TARGET_SIZE[1]:
                    os.remove(img_path)
                    continue
                img = img.convert('RGB')
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                img.save(img_path)
        except Exception as e:
            print(e)

for folder in os.listdir(OUTPUT_DIR):
    resize_images(os.path.join(OUTPUT_DIR, folder))

print("done resizing!")
