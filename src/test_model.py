import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/pythia-160m"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

print("\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))

if device == "cuda":
    print("\nVRAM used:",
          round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")