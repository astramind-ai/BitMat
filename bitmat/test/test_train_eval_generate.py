import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitmat import convert_hf_model
from bitmat import Auto158ModelForCausalLM
from datasets import load_dataset


# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model_orig = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Convert model to a format suitable for use with BitMat
model_bit = convert_hf_model(model_orig)
model_bit.save_pretrained('gemma-2b', safe_serialization=True)

# Re-load the model using custom classes
model = Auto158ModelForCausalLM.from_pretrained('./gemma-2b')

# Set up model for inference and move to CUDA
model.cuda()
model.eval()

# Tokenize a sample text and generate text
tokenized_text = tokenizer("write a short poem", return_tensors='pt').to('cuda')
generated_text = model.generate(tokenized_text['input_ids'], max_length=50)
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print("Generated Text Before Training:", decoded_text)

# Load Alpaca dataset from Hugging Face
dataset = load_dataset("tatsu-lab/alpaca")

# Prepare dataset for training
texts = [example['text'] for example in dataset['train']]  # Assuming 'train' split is used
tokenizer.model_max_length = 1024
tokenized_ds = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Parameters
batch_size = 1  # Adjust batch size here
total_training_steps = 5000
num_examples = len(tokenized_ds['input_ids'])
total_batches = total_training_steps
epochs = total_batches * batch_size // num_examples + (total_batches * batch_size % num_examples > 0)

# Define loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

current_step = 0

for epoch in range(epochs):
    for i in range(0, num_examples, batch_size):
        if current_step >= total_training_steps:
            break
        batch_end = min(i + batch_size, num_examples)
        input_ids = tokenized_ds['input_ids'][i:batch_end].to('cuda')
        attention_mask = tokenized_ds['attention_mask'][i:batch_end].to('cuda')
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Step {current_step+1}/{total_training_steps}, Loss: {loss.item()}")
        current_step += 1

        # Save checkpoint after the 5000th training step
        if current_step == total_training_steps:
            checkpoint_path = f"checkpoint_step_{current_step}.pt"
            torch.save({
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {current_step} to {checkpoint_path}")
            break

    if current_step >= total_training_steps:
        break

print("Training session completed.")
torch.cuda.empty_cache()  # Clear CUDA cache after training

model.eval()  # Set model back to evaluation mode
post_train_generated_text = model.generate(tokenized_text['input_ids'], max_length=50)
post_train_decoded_text = tokenizer.decode(post_train_generated_text[0], skip_special_tokens=True)
print("Generated Text After Training:", post_train_decoded_text)


# Load the checkpoint
checkpoint_path = "./checkpoint_step_5000.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
starting_step = checkpoint['step']
loss = checkpoint['loss']

print(f"Resumed from checkpoint '{checkpoint_path}' at step {starting_step} with loss {loss}.")