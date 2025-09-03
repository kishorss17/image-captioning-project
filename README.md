# image-captioning-project
An image captioning system using a ResNet18 encoder and LSTM decoder trained on MS COCO data. It generates natural language captions describing image content, enabling applications like accessibility for visually impaired users, automated image tagging, and content summarization with high accuracy.
Image Captioning Project
Project Overview
This project implements an image captioning system that generates natural language descriptions for images. It uses a ResNet18 encoder to extract image features and an LSTM decoder to create captions. The model is trained on a mini subset of the MS COCO dataset.

Features
Image encoding with pre-trained ResNet18

Caption generation with LSTM decoder

Data preprocessing and vocabulary building

Training and inference pipeline

Download script for mini COCO dataset subset

Optional Streamlit dashboard for generating captions and segmenting images

Dataset
The project uses a small subset of the MS COCO 2017 dataset, which can be downloaded by running:

text
python download_mini_dataset.py
This script downloads a small set of images and their annotations automatically.

Requirements
See requirements.txt for Python dependencies.

To install all requirements, run:

text
pip install -r requirements.txt
How to Run
Training
Run the training script:

text
python train.py
Inference / Dashboard
To run the interactive dashboard for captioning and segmentation (requires Streamlit):

text
streamlit run src/dashboard.py
Upload an image and generate captions or segmentation maps easily through the web interface.

Project Structure
text
├── data/                     # Dataset images and annotations (downloaded via script)
├── src/
│   ├── preprocessing.py      # Dataset and data loader code
│   ├── model_training.py     # Model architecture and training functions
│   ├── dashboard.py          # Streamlit app for inference
├── train.py                  # Training script
├── download_mini_dataset.py  # Script to download mini dataset
├── requirements.txt          # Python library requirements
Notes
Images are not included in the repository due to size. Use the download script.

The model checkpoint is saved as models/image_captioning_model.pth after training.
