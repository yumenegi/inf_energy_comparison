import os
from tqdm import tqdm
from PIL import Image

FOLDER_MAPPING = {
    'goose': 'goose',
    'wood_rabbit': 'rabbit',
    'fox_squirrel': 'squirrel',
    'basketball': 'basketball',
    'cassette': 'cassette',
    'desk': 'desk',
    'hammer': 'hammer',
    'jersey': 't shirt',
    'minivan': 'minivan',
    'oscilloscope': 'oscilloscope',
    'pillow': 'pillow',
    'printer': 'printer',
    'monitor': 'screen',
    'submarine': 'submarine',
    'switch': 'switch',
    'water_bottle': 'water bottle',
    'wooden_spoon': 'wooden spoon',
    'hotdog': 'hotdog',
    'orange': 'orange',
    'corn': 'corn'
}

FETCHED_DIR = './fetched' 
TARGET_SIZE = (80, 80)

for orig_folder, target_class in FOLDER_MAPPING.items():
    input_folder = os.path.join(FETCHED_DIR, orig_folder)
    output_folder = os.path.join("./data", target_class.replace(' ', '_'))
    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc=f"processing {orig_folder}"):
        img_path = os.path.join(input_folder, img_name)
        with Image.open(img_path) as img:
            if img.width < TARGET_SIZE[0] or img.height < TARGET_SIZE[1]:
                continue
            img = img.convert('RGB')
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_folder, img_name))
