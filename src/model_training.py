import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # Using smaller ResNet18
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.cnn(images)
        features = self.relu(features)
        features = self.dropout(features)
        return features


class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        # Concatenate image features as the first input to LSTM sequence
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


def train_model(data_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0
        
        for images, captions, caption_lengths in progress_bar:
            # Prepare inputs and targets by removing special tokens as needed
            targets = captions[:, 1:]       # Remove <start> token from targets
            inputs = captions[:, :-1]       # Remove <end> token from inputs
            
            outputs = model(images, inputs)  # Forward pass
            
            # Compute loss, flatten outputs and targets
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses


# Optional: Function to get a pre-trained segmentation model (not related to captioning)
def get_segmentation_model():
    model = models.segmentation.fcn_resnet50(pretrained=True)  # Lightweight segmentation model
    return model
