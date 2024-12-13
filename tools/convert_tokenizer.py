import json
import os
import base64
import regex as re
from collections import OrderedDict

def convert_to_tiktoken(input_json_path, output_tiktoken_path):
    """Convert HuggingFace tokenizer.json to tiktoken format"""
    
    # Read the input tokenizer.json
    with open(input_json_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Create a dictionary to track tokens and indices
    token_map = OrderedDict()
    used_indices = set()
    
    def safe_encode_token(token):
        try:
            # For special tokens, use the token directly
            if token.startswith('<') and token.endswith('>'):
                return base64.b64encode(token.encode('utf-8')).decode('ascii')
            
            # For regular tokens, encode each byte separately
            token_bytes = token.encode('utf-8')
            return base64.b64encode(token_bytes).decode('ascii')
        except Exception as e:
            print(f"Warning: Failed to encode token {token}: {e}")
            return None
    
    # First add byte-level tokens (0-255)
    for i in range(256):
        byte_token = bytes([i])
        b64_token = base64.b64encode(byte_token).decode('ascii')
        token_map[b64_token] = i
        used_indices.add(i)
    
    next_id = 256
    
    # Add special tokens
    for token in tokenizer_data.get('added_tokens', []):
        if token.get('special', False):
            content = token['content']
            idx = token['id']
            if idx < 256:  # Skip if conflicts with byte tokens
                continue
            b64_token = safe_encode_token(content)
            if b64_token and b64_token not in token_map:
                token_map[b64_token] = idx
                used_indices.add(idx)
                next_id = max(next_id, idx + 1)
    
    # Add vocabulary tokens
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    for token, idx in vocab.items():
        try:
            idx = int(idx)
            if idx < 256:  # Skip if conflicts with byte tokens
                continue
            b64_token = safe_encode_token(token)
            if b64_token and b64_token not in token_map:
                token_map[b64_token] = idx
                used_indices.add(idx)
                next_id = max(next_id, idx + 1)
        except (ValueError, TypeError):
            continue
    
    # Sort by index
    sorted_tokens = sorted(token_map.items(), key=lambda x: x[1])
    
    # Create tiktoken entries
    tiktoken_entries = [f"{token}\t{idx}" for token, idx in sorted_tokens]
    
    # Write the tiktoken file
    with open(output_tiktoken_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tiktoken_entries))
    
    print(f"Created tokenizer with {len(token_map)} tokens")
    print(f"Index range: {min(token_map.values())} to {max(token_map.values())}")
    print(f"Number of unique indices: {len(set(token_map.values()))}")
    print(f"Number of unique tokens: {len(set(token_map.keys()))}")

if __name__ == "__main__":
    # Paths
    model_path = "checkpoints/fish-speech-1.4"
    input_json = os.path.join(model_path, "tokenizer.json")
    output_tiktoken = os.path.join(model_path, "tokenizer.tiktoken")
    
    # Convert
    convert_to_tiktoken(input_json, output_tiktoken)
    
    # Verify the output file
    with open(output_tiktoken, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        indices = [int(line.strip().split('\t')[1]) for line in lines]
        tokens = [line.strip().split('\t')[0] for line in lines]
        print("\nVerification:")
        print(f"Unique tokens in file: {len(set(tokens))}")
        print(f"Unique indices in file: {len(set(indices))}")
        print(f"Total entries in file: {len(lines)}")