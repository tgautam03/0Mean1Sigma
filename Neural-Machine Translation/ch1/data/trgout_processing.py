# Output for Decoder
def output_trg_processing(tokenized_text, max_seq_len):
    # Padding or trimming to fit the max sequence length
    if len(tokenized_text) < max_seq_len:
        # Add EOS token at the end
        tokenized_text += [2]
        left = max_seq_len - len(tokenized_text)
        padding = [0] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:max_seq_len-1]
        # Add EOS token at the end
        tokenized_text += [2]
        
    return tokenized_text