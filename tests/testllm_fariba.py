import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm

# Step 1: Load GPT-2 tokenizer and add custom log templates
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

log_templates = [
    "Error encountered in module X",
    "Error encountered in",
    "Unexpected behavior in network communication",
    "System rebooted successfully",
    "Segmentation fault in memory allocation"
]

# Add log templates as tokens to the tokenizer
tokenizer.add_tokens(log_templates)

# Set padding token to EOS
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Prepare log sequences (you may add your actual log data here)
sequences = [
    "Error encountered in module X The weather is great today. I am working hard.",
    "The system rebooted successfully after the error."
]

# Tokenize sequences
tokenized_sequences = tokenizer(
    sequences,  # List of sentences
    truncation=True,
    padding=True,  # Padding all sequences to the same length
    max_length=128,  # Set max length for input sequences
    return_tensors="pt"  # Return PyTorch tensors
)

# Step 3: Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize the model's embeddings to include the new tokens
model.resize_token_embeddings(len(tokenizer))

# Step 4: Prepare Dataset and DataLoader for training
class LogSequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = tokenized_sequences['input_ids']
        self.attention_mask = tokenized_sequences['attention_mask']
        
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Create the dataset and dataloader
dataset = LogSequenceDataset(tokenized_sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Enable gradient updates on embeddings
model.get_input_embeddings().requires_grad_(True)

# Step 7: Train the model (Continual Pretraining)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3  # Adjust based on your training needs

for epoch in range(epochs):
    model.train()
    loop = tqdm(dataloader, leave=True)
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (language modeling task)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update the model's parameters
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

print("Fine-tuning completed and model saved.")