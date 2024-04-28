import sentencepiece as spm
import numpy as np

# Input to the encoder
def src_processing(tokenized_text:list, max_seq_len:int):
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

# Input to Decoder
def input_trg_processing(tokenized_text:list, max_seq_len:int):
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

# Output for Decoder
def output_trg_processing(tokenized_text:list, max_seq_len:int):
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