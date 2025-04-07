import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
import readline
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

def setup_history(history_file):
    # Set up command history
    histfile = os.path.expanduser(history_file)
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(histfile), exist_ok=True)
    
    # Save history on exit
    import atexit
    atexit.register(readline.write_history_file, histfile)

def main(args):
    # Determine history file path
    history_file = args.history_file
    
    # Set up command history
    setup_history(history_file)
    
    # Model initialization
    model_path = os.path.expanduser(args.model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map='auto'
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map='auto'
    )
    processor.chat_template = processor.tokenizer.chat_template
    
    # Initialize conversation
    image = None
    conversation_started = False
    
    print(f"\nMolmo Chat Interface - Model: {os.path.basename(model_path)}")
    print(f"History file: {history_file}")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'load_image [path]' to load an image (only before conversation starts)")
    print("Type 'clear' to start a new conversation")
    print("Use up/down arrow keys to navigate command history\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit"]:
            break
            
        if user_input.lower() == "clear":
            image = None
            conversation_started = False
            print("Conversation cleared")
            continue
            
        if user_input.lower().startswith("load_image "):
            if conversation_started:
                print("Cannot load image after conversation has started. Please 'clear' first.")
                continue
                
            image_path = user_input[11:].strip()
            try:
                image = Image.open(os.path.expanduser(image_path)).convert('RGB')
                print(f"Image loaded: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
        
        if image is None:
            print("Please load an image first using 'load_image [path]'")
            continue
        
        # Process input with image
        inputs = processor.process(
            images=[image],
            text=user_input,
            return_tensors='pt',
        )
        
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate response
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=args.max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
        
        # Decode and display response
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        conversation_started = True
        print(f"\nAssistant: {generated_text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='samarthm44/SCRAMBLe-Molmo-7B-D-0924', help="Path to the Molmo model")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--history-file", type=str, default='./.molmo_chat_history', 
                        help="Path to command history file (default: ./.molmo_chat_history)")
    args = parser.parse_args()
    
    main(args)