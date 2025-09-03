import torch
import torch.nn as nn
from src.preprocessing import get_data_loader
from src.model_training import ImageCaptioningModel

# Hyperparameters (smaller for our mini dataset)
embed_size = 128
hidden_size = 256
num_layers = 1
num_epochs = 5
batch_size = 2
learning_rate = 0.001

print("Loading data...")
data_loader, vocab, reverse_vocab = get_data_loader(
    'data/images/train2017', 
    'data/annotations/captions_mini2017.json', 
    batch_size=batch_size
)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Initialize model, loss, and optimizer
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")

def train_model_with_trimmed_output(data_loader, model, criterion, optimizer, num_epochs):
    model.train()
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, captions, lengths in data_loader:
            optimizer.zero_grad()
            
            # Remove <end> token from input captions to predict next word; keep same length for targets
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            
            outputs = model(images, inputs)  # Model output shape: (batch, seq_len, vocab_size)
            
            max_caption_length = targets.size(1)
            outputs = outputs[:, :max_caption_length, :]  # Trim outputs to targets length
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        all_losses.append(avg_loss)
    return all_losses

losses = train_model_with_trimmed_output(data_loader, model, criterion, optimizer, num_epochs)

# Save model and vocabulary
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'reverse_vocab': reverse_vocab,
    'embed_size': embed_size,
    'hidden_size': hidden_size,
    'vocab_size': vocab_size,
    'num_layers': num_layers
}, 'models/image_captioning_model.pth')

print("Training completed! Model saved.")
