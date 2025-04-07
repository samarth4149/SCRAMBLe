import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import torch
from PIL import Image
import readline  # Add this import for command history

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def setup_history(history_file):
    # Set up command history
    histfile = os.path.expanduser(history_file)
    try:
        readline.read_history_file(histfile)
        # Default history len is -1 (infinite), which may grow unruly
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
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name or get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    
    # Initialize conversation
    conv = conv_templates[args.conv_mode].copy()
    image = None
    image_tensor = None
    conversation_started = False
    
    print(f"\nLLaVA Chat Interface - Model: {model_name}")
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
            conv = conv_templates[args.conv_mode].copy()
            image = None
            image_tensor = None
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
                image_tensor = process_images([image], image_processor, model.config)[0]
                print(f"Image loaded: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
        
        # Prepare input with image token if image is loaded
        if image is not None and not conversation_started:
            # Only add image token to the first message in the conversation
            if model.config.mm_use_im_start_end:
                user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_input
            else:
                user_input = DEFAULT_IMAGE_TOKEN + '\n' + user_input
        
        # Add user message to conversation
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda() if image_tensor is not None else None,
                image_sizes=[image.size] if image is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        
        # Decode and display response
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Update conversation with assistant's response
        conv.messages[-1][-1] = outputs
        conversation_started = True
        
        print(f"\nAssistant: {outputs}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="samarthm44/SCRAMBLe-llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--history-file", type=str, default='./.chat_history', 
                        help="Path to command history file (default: .chat_history)")
    args = parser.parse_args()
    
    main(args)
