from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
from datasets import Dataset
import os

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16
)

# Set up model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model with proper configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Check GPU availability and print status
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("No GPU available, using CPU instead")
    exit()

# Prepare the model for kbit training - THIS IS CRITICAL FOR GRADIENT COMPUTATION
model = prepare_model_for_kbit_training(model)

# Configure LoRA with correct target modules for DeepSeek
# Find the actual attention module names in the model
target_modules = []
for name, module in model.named_modules():
    if 'attn' in name and any(substr in name for substr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        parent_name = name.rsplit('.', 1)[0]
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if f"{parent_name}.{proj}" not in target_modules:
                target_modules.append(f"{parent_name}.{proj}")

print(f"Target modules: {target_modules}")

if not target_modules:
    print("Warning: No target modules found. Falling back to default target modules.")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Create PEFT model
model = get_peft_model(model, lora_config)

# Verify trainable parameters exist and print detailed info
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# Print more detailed information about trainable vs non-trainable parameters
total_params = 0
trainable_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"Trainable: {name} - {param.numel()} parameters")

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}%)")

# If no trainable parameters, check the model structure
if trainable_params == 0:
    print("WARNING: No trainable parameters found!")
    # Try forcing requires_grad for LoRA modules
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            print(f"Forced training for: {name}")
    
    # Check again after forcing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After forcing, trainable parameters: {trainable_params}")

# Prepare training data with more examples
training_data = [
    {"text": "Question: What is innovation-speak? Answer: Innovation-speak refers to the public discourse and rhetoric surrounding innovation, often criticized for overuse and lack of meaningful content."},
    {"text": "Question: Tell me about innovation-speak. Answer: Innovation-speak is a term used to describe the excessive and often superficial talk about innovation, which can obscure other important aspects of technology and society."},
    {"text": "Question: Describe innovation-speak. Answer: Innovation-speak is the cultural obsession with innovation as a buzzword, often leading to the devaluation of maintenance and other essential non-innovative work."},
    {"text": "Question: What are the effects of innovation-speak? Answer: Innovation-speak can lead to the devaluation of maintenance work, create false self-images among students, and obscure the importance of non-innovative roles in society."},
    {"text": "Question: Is innovation-speak harmful? Answer: Yes, innovation-speak can be harmful as it often devalues essential maintenance work and creates unrealistic expectations about the role of innovation in society."},
    {"text": "Question: What is the ethics of care? Answer: The ethics of care is a moral framework that emphasizes interdependence, responsiveness, and attention to the vulnerable, often contrasting with the individualistic focus of innovation-speak."},
    {"text": "Question: Tell me about the ethics of care. Answer: The ethics of care is rooted in feminist theory and focuses on the importance of relationships, care, and maintenance in ethical decision-making, particularly in fields like engineering."},
    {"text": "Question: Describe the ethics of care. Answer: The ethics of care is an ethical framework that prioritizes care, maintenance, and responsiveness to the needs of others, offering an alternative to the innovation-centric focus in engineering education."},
    {"text": "Question: How does the ethics of care apply to engineering? Answer: The ethics of care in engineering emphasizes the importance of maintaining and caring for existing technological systems, rather than solely focusing on innovation and new designs."},
    {"text": "Question: Is the ethics of care important in engineering education? Answer: Yes, the ethics of care is important in engineering education as it provides a more holistic view of technology, emphasizing maintenance and care over innovation."},
    {"text": "Question: What is the role of maintenance in engineering? Answer: Maintenance in engineering involves the upkeep and repair of existing technological systems, which is essential for the reliable functioning of society."},
    {"text": "Question: Tell me about the role of maintenance in engineering. Answer: Maintenance is a critical aspect of engineering, as most engineers work on maintaining and overseeing existing systems rather than designing new ones."},
    {"text": "Question: Describe the importance of maintenance in engineering. Answer: Maintenance is crucial in engineering because it ensures the continuous and reliable operation of technological systems, which are vital for everyday life."},
    {"text": "Question: Why is maintenance important in engineering? Answer: Maintenance is important in engineering because it ensures the longevity and reliability of technological systems, which are often more critical than innovation in sustaining society."},
    {"text": "Question: How can engineering education be reformed? Answer: Engineering education can be reformed by incorporating the ethics of care, focusing on maintenance, and balancing innovation with the need for upkeep and care of existing systems."},
    {"text": "Question: What are the criticisms of current engineering education? Answer: Current engineering education is criticized for its overemphasis on innovation and design, often neglecting the importance of maintenance and the ethics of care."},
    {"text": "Question: How does innovation-speak affect engineering students? Answer: Innovation-speak can create unrealistic expectations for engineering students, leading to disillusionment when they enter non-innovative but essential maintenance roles."},
    {"text": "Question: What is the Big Beacon Movement? Answer: The Big Beacon Movement is a reform initiative in engineering education that advocates for disruptive change and innovation, often aligning with the rhetoric of innovation-speak."},
    {"text": "Question: Tell me about the Big Beacon Movement. Answer: The Big Beacon Movement seeks to transform engineering education by promoting innovation and entrepreneurship, often at the expense of traditional maintenance-focused roles."},
    {"text": "Question: Describe the Big Beacon Movement. Answer: The Big Beacon Movement is a reform effort in engineering education that emphasizes innovation and disruption, often criticized for overlooking the importance of maintenance and care."},
    {"text": "Question: What is the Maintainers network? Answer: The Maintainers network is a global research group focused on the study of maintenance, repair, and the labor that sustains technological systems."},
    {"text": "Question: Tell me about the Maintainers network. Answer: The Maintainers network is an interdisciplinary research group that studies the importance of maintenance and repair in sustaining technological and social systems."},
    {"text": "Question: Describe the Maintainers network. Answer: The Maintainers network is a research initiative that highlights the critical role of maintenance and repair in technology, often contrasting with the innovation-centric focus of modern discourse."},
    {"text": "Question: What is the significance of the Maintainers network? Answer: The Maintainers network is significant because it brings attention to the often-overlooked importance of maintenance and repair in sustaining technological systems and society."},
    {"text": "Question: How does the Maintainers network view innovation? Answer: The Maintainers network critiques the overemphasis on innovation and advocates for a more balanced view that includes the importance of maintenance and care in technology."}
]

# Create and process dataset
def tokenize_function(examples):
    # Make sure to set return_tensors=None here
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors=None
    )
    
    # Set up labels for causal LM
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# Create dataset
dataset = Dataset.from_list(training_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create a proper data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=25,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=10,
    save_total_limit=2,
    learning_rate=1e-4,
    warmup_steps=5, 
    logging_steps=1,  # Log every step to see progress
    fp16=True,
    remove_unused_columns=False,
    # Explicitly set these to avoid conflicts
    gradient_checkpointing=True,
    optim="adamw_torch",  # Use PyTorch's optimizer
)

# Initialize trainer with proper data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Add a function to verify inputs before training
def verify_batch(batch):
    """Verify a batch has proper tensors that can receive gradients."""
    print("\nVerifying batch for gradient computation:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, requires_grad={v.requires_grad}")
    
    # Try creating a dummy forward pass to check gradient flow
    try:
        print("\nTesting gradient flow with dummy forward pass...")
        model.train()
        outputs = model(**{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)})
        loss = outputs.loss
        print(f"Loss computed: {loss.item()}, requires_grad={loss.requires_grad}")
        
        # Test if loss can receive gradients
        if loss.requires_grad:
            print("Testing gradient computation...")
            loss.backward()
            print("✓ Gradient computation successful!")
        else:
            print("✗ Loss doesn't require gradients!")
        
        return True
    except Exception as e:
        print(f"Error in dummy forward pass: {str(e)}")
        return False

# Run verification on a sample batch before training
sample_batch = data_collator([tokenized_dataset[i] for i in range(min(2, len(tokenized_dataset)))])
verification_success = verify_batch(sample_batch)

# Train the model with extended error handling
print("\nStarting training...")
try:
    if verification_success:
        trainer.train()
        print("Training completed!")
    else:
        print("Skipping training due to verification failure.")
except Exception as e:
    print(f"Training error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Try one last approach: manual training loop for one batch
    print("\nAttempting manual training loop for diagnosis...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    sample_batch = data_collator([tokenized_dataset[i] for i in range(min(2, len(tokenized_dataset)))])
    try:
        # Move tensors to device
        inputs = {k: v.to(device) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Manual forward pass loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        print("Manual backward pass successful")
        
        # Update parameters
        optimizer.step()
        print("Manual optimization step successful")
        
        print("Manual training loop succeeded - problem may be in the Trainer configuration")
    except Exception as inner_e:
        print(f"Manual training loop failed: {str(inner_e)}")
        traceback.print_exc()

# Move the test function outside the conditional block
def generate_text(prompt, max_length=100):
    formatted_prompt = f"Question: {prompt} Answer:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save the model regardless of training state
try:
    output_dir = "./trained_lora_model2"
    print(f"Saving LoRA model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA model saved to {output_dir}")
except Exception as e:
    print(f"Error saving model: {str(e)}")
    import traceback
    traceback.print_exc()

# Always run test examples regardless of training success
print("\n=== Testing Model Generation ===")
test_prompts = [
    "Describe the concept of Ethics of Care?",
    "What is innovation-speak?",
    "How does maintenance relate to engineering?"
]

for test_prompt in test_prompts:
    print(f"\nTesting model with prompt: {test_prompt}")
    generated_text = generate_text(test_prompt)
    print(f"Generated response: {generated_text}")
