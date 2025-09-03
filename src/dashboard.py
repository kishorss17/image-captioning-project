# src/dashboard.py
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    # For captioning - using BLIP which works well with small datasets
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # For segmentation
    segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    segmentation_model.eval()
    
    return processor, caption_model, segmentation_model

processor, caption_model, segmentation_model = load_models()

st.title("🖼️ Image Captioning & Segmentation")
st.markdown("Upload an image to generate captions and perform segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Captioning")
        if st.button("Generate Caption"):
            # Generate caption using BLIP model
            inputs = processor(image, return_tensors="pt")
            out = caption_model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.success(f"**Generated Caption:** {caption}")
    
    with col2:
        st.subheader("Image Segmentation")
        if st.button("Perform Segmentation"):
            # Preprocess image for segmentation
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Perform segmentation
            with torch.no_grad():
                output = segmentation_model(input_tensor)['out'][0]
            
            # Get the segmentation mask
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Display the mask
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(output_predictions)
            ax.axis('off')
            st.pyplot(fig)

else:
    st.info("⬆️ Please upload an image to get started.")