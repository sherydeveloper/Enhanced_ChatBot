from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    attention_mask = inputs['attention_mask']
    input_ids = inputs['input_ids']
    
    # Generate response with adjusted parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=255,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Adjust temperature for randomness
            top_k=50,  # Use top-k sampling
            top_p=0.9  # Use top-p sampling (nucleus sampling)
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chatbot():
    print("Hi! I am your chatbot powered by GPT-2. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = get_response(user_input)
        print("ChatBot: " + response)

if __name__ == "__main__":
    chatbot()
