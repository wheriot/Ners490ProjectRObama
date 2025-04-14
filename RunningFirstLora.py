from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("No GPU available, using CPU instead. This might be slow.")

# Path to your saved LoRA model
lora_model_path = "./trained_lora_model"
output_dir = "./merged_model"
os.makedirs(output_dir, exist_ok=True)

# Load base model with 8-bit quantization for efficiency
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading base model: {base_model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# Load the LoRA adapter
print(f"Loading LoRA adapter from: {lora_model_path}")
model = PeftModel.from_pretrained(model, lora_model_path)

# Merge LoRA weights with base model
print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {output_dir}")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nModel has been merged and saved. To convert to GGUF format, you'll need to:")
print("1. Install llama.cpp: https://github.com/ggerganov/llama.cpp")
print("2. Run the following command:")
print(f"python -m llama_cpp.convert_hf_to_gguf {output_dir} --outfile merged_model.gguf")

# Set to evaluation mode
model.eval()

def generate_response(prompt, max_length=200):
    """Generate a response for the given prompt"""
    # Format the prompt as in training
    formatted_prompt = f"Question: {prompt} Answer:"
    print(f"\nPrompt: {formatted_prompt}")
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interactive loop
print("\n" + "="*50)
print("Model loaded and ready! Type 'exit' to quit.")
print("="*50)

while True:
    user_input = input("\nEnter your question: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break
    
    try:
        response = generate_response(user_input)
        print("\nResponse:", response)
    except Exception as e:
        print(f"Error generating response: {str(e)}")
    
print("\nThank you for using the model!")