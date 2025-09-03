# download_mini_dataset.py
import os
import requests
import json
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
    print(f"Downloaded: {filename}")

# Create directories
os.makedirs('data/images/train2017', exist_ok=True)
os.makedirs('data/images/val2017', exist_ok=True)
os.makedirs('data/annotations', exist_ok=True)

# Download sample images (20 images - 10 train, 10 val)
print("Downloading sample images...")
image_urls = [
    # Training images
    "http://images.cocodataset.org/train2017/000000000009.jpg",
    "http://images.cocodataset.org/train2017/000000000025.jpg",
    "http://images.cocodataset.org/train2017/000000000030.jpg",
    "http://images.cocodataset.org/train2017/000000000034.jpg",
    "http://images.cocodataset.org/train2017/000000000036.jpg",
    "http://images.cocodataset.org/train2017/000000000042.jpg",
    "http://images.cocodataset.org/train2017/000000000057.jpg",
    "http://images.cocodataset.org/train2017/000000000061.jpg",
    "http://images.cocodataset.org/train2017/000000000071.jpg",
    "http://images.cocodataset.org/train2017/000000000074.jpg",
    
    # Validation images
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
    "http://images.cocodataset.org/val2017/000000000785.jpg",
    "http://images.cocodataset.org/val2017/000000000802.jpg",
    "http://images.cocodataset.org/val2017/000000000991.jpg",
    "http://images.cocodataset.org/val2017/000000001000.jpg",
    "http://images.cocodataset.org/val2017/000000001087.jpg"
]

for i, url in enumerate(tqdm(image_urls, desc="Downloading images")):
    if 'train2017' in url:
        filename = f'data/images/train2017/{url.split("/")[-1]}'
    else:
        filename = f'data/images/val2017/{url.split("/")[-1]}'
    
    download_file(url, filename)

# Create mini annotations
print("Creating mini annotations...")
mini_annotations = {
    "info": {
        "description": "Mini COCO Dataset for Image Captioning",
        "version": "1.0",
        "year": 2024
    },
    "images": [
        {"id": 9, "file_name": "000000000009.jpg", "width": 640, "height": 427},
        {"id": 25, "file_name": "000000000025.jpg", "width": 640, "height": 426},
        {"id": 30, "file_name": "000000000030.jpg", "width": 500, "height": 375},
        {"id": 34, "file_name": "000000000034.jpg", "width": 500, "height": 375},
        {"id": 36, "file_name": "000000000036.jpg", "width": 500, "height": 375},
        {"id": 42, "file_name": "000000000042.jpg", "width": 500, "height": 375},
        {"id": 57, "file_name": "000000000057.jpg", "width": 500, "height": 375},
        {"id": 61, "file_name": "000000000061.jpg", "width": 500, "height": 375},
        {"id": 71, "file_name": "000000000071.jpg", "width": 500, "height": 375},
        {"id": 74, "file_name": "000000000074.jpg", "width": 500, "height": 375},
        {"id": 139, "file_name": "000000000139.jpg", "width": 640, "height": 427},
        {"id": 285, "file_name": "000000000285.jpg", "width": 640, "height": 427},
        {"id": 632, "file_name": "000000000632.jpg", "width": 640, "height": 427},
        {"id": 724, "file_name": "000000000724.jpg", "width": 640, "height": 427},
        {"id": 776, "file_name": "000000000776.jpg", "width": 640, "height": 427},
        {"id": 785, "file_name": "000000000785.jpg", "width": 640, "height": 427},
        {"id": 802, "file_name": "000000000802.jpg", "width": 640, "height": 427},
        {"id": 991, "file_name": "000000000991.jpg", "width": 640, "height": 427},
        {"id": 1000, "file_name": "000000001000.jpg", "width": 640, "height": 427},
        {"id": 1087, "file_name": "000000001087.jpg", "width": 640, "height": 427}
    ],
    "annotations": [
        {"id": 1, "image_id": 9, "caption": "A street with cars and buildings in a city"},
        {"id": 2, "image_id": 25, "caption": "A person riding a skateboard on the street"},
        {"id": 3, "image_id": 30, "caption": "A cat sitting on a wooden chair"},
        {"id": 4, "image_id": 34, "caption": "A group of people playing frisbee in a field"},
        {"id": 5, "image_id": 36, "caption": "A red bicycle parked on the street"},
        {"id": 6, "image_id": 42, "caption": "A busy city intersection with traffic"},
        {"id": 7, "image_id": 57, "caption": "A close-up of a colorful bird on a branch"},
        {"id": 8, "image_id": 61, "caption": "A person sitting on a bench in the park"},
        {"id": 9, "image_id": 71, "caption": "A delicious pizza on a wooden table"},
        {"id": 10, "image_id": 74, "caption": "A beautiful sunset over the ocean"},
        {"id": 11, "image_id": 139, "caption": "A dog running through green grass"},
        {"id": 12, "image_id": 285, "caption": "A bowl of fresh salad with vegetables"},
        {"id": 13, "image_id": 632, "caption": "A city skyline with tall buildings"},
        {"id": 14, "image_id": 724, "caption": "A person riding a horse in a field"},
        {"id": 15, "image_id": 776, "caption": "A close-up of a colorful flower"},
        {"id": 16, "image_id": 785, "caption": "A group of friends having a picnic"},
        {"id": 17, "image_id": 802, "caption": "A vintage car parked on the street"},
        {"id": 18, "image_id": 991, "caption": "A beautiful mountain landscape"},
        {"id": 19, "image_id": 1000, "caption": "A person cooking in a kitchen"},
        {"id": 20, "image_id": 1087, "caption": "A child playing with a ball in the park"}
    ]
}

with open('data/annotations/captions_mini2017.json', 'w') as f:
    json.dump(mini_annotations, f, indent=2)

print("Mini dataset created successfully!")
print("Training images:", len(os.listdir('data/images/train2017')))
print("Validation images:", len(os.listdir('data/images/val2017')))