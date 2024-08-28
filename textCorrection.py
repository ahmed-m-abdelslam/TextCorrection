import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def ocr_correction(prompt, max_new_tokens=600):
    # Create the prompt for the model
    prompt = f"""### Text ###\n{prompt}\n\n\n### Correction ###\n"""

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text with the model
    output = model.generate(input_ids,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                            top_k=3)

    # Decode the output and strip the prompt part
    corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the leading part of the output until after 'Correction:'
    corrected_text = corrected_text.split("\n")[-1].strip()
    
    return corrected_text

# Load pre-trained model and tokenizer
model_name = "PleIAs/OCRonos-Vintage"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example usage
prompt = "lok at th ski and its stars i wold lik to have this meel"
ocr_result = ocr_correction(prompt)
print(ocr_result)