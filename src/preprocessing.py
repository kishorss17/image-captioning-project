import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from nltk.tokenize import word_tokenize
import nltk
import random

nltk.download('punkt')

class MiniCocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, max_caption_length=20):
        self.image_dir = image_dir
        self.transform = transform
        self.max_caption_length = max_caption_length
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create mapping from image id to image info
        self.image_id_to_info = {img['id']: img for img in self.annotations['images']}
        
        # Create mapping from image id to captions
        self.image_id_to_captions = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_captions:
                self.image_id_to_captions[image_id] = []
            self.image_id_to_captions[image_id].append(ann['caption'])
        
        # Create list of image ids that have captions
        self.image_ids = list(self.image_id_to_captions.keys())
        
        # Build vocabulary
        self.build_vocabulary()
        
    def build_vocabulary(self):
        self.vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.reverse_vocab = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        
        word_freq = {}
        for image_id in self.image_ids:
            for caption in self.image_id_to_captions[image_id]:
                tokens = word_tokenize(caption.lower())
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
        
        # Add words to vocabulary
        for word, freq in word_freq.items():
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word
        
        self.vocab_size = len(self.vocab)
    
    def numericalize(self, caption):
        tokens = word_tokenize(caption.lower())
        numericalized = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        numericalized = [self.vocab['<start>']] + numericalized[:self.max_caption_length-2] + [self.vocab['<end>']]
        return numericalized
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        attempts = 0
        max_attempts = len(self.image_ids)
        while attempts < max_attempts:
            image_id = self.image_ids[idx]
            image_info = self.image_id_to_info[image_id]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                caption = random.choice(self.image_id_to_captions[image_id])
                numericalized_caption = self.numericalize(caption)
                return image, torch.tensor(numericalized_caption)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                idx = random.randint(0, len(self.image_ids) - 1)
                attempts += 1
        raise FileNotFoundError(f"No valid image found after {max_attempts} attempts.")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    caption_lengths = [len(cap) for cap in captions]
    max_length = max(caption_lengths)
    padded_captions = torch.zeros(len(captions), max_length).long()
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions, caption_lengths

def get_data_loader(image_dir, annotation_file, batch_size=4, shuffle=True):
    dataset = MiniCocoDataset(image_dir, annotation_file, transform=transform)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return data_loader, dataset.vocab, dataset.reverse_vocab
