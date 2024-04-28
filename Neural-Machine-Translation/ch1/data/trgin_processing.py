# Input to Decoder
def input_trg_processing(tokenized_text, max_seq_len):
    # Add BOS token at the start
    tokenized_text = [1] + tokenized_text
    # Padding or trimming to fit the max sequence length
    if len(tokenized_text) < max_seq_len:
        left = max_seq_len - len(tokenized_text)
        padding = [0] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:max_seq_len]
        
    return tokenized_text