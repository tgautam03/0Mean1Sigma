import torch
import sentencepiece as spm

from processing import src_processing
from transformer import Transformer

class Translate:
    def __init__(self, trained_model, device="cpu", model_details={"emb":256, "heads":16, "max_seq_len":256, "num_layers":4}):
        self.max_seq_len = model_details["max_seq_len"]
        self.device = device
        # Loading Tokenizers
        self.shakespeare = spm.SentencePieceProcessor(model_file="../trained_models/tokenizer/shakespeare_en.model")
        self.modern = spm.SentencePieceProcessor(model_file="../trained_models/tokenizer/modern_en.model")

        # Loading NN
        self.model = Transformer(emb=model_details["emb"], heads=model_details["heads"], 
                            max_seq_len=model_details["max_seq_len"], 
                            src_vocab_len=self.shakespeare.vocab_size(), trg_vocab_len=self.modern.vocab_size(),
                            num_layers=model_details["num_layers"], device=device).to(device)
        self.model.load_state_dict(torch.load(trained_model))

    def __call__(self, input):
        max_seq_len = self.max_seq_len
        # Processing input
        self.text = self.shakespeare.Encode(input, out_type=str)
        tokenized_input = self.shakespeare.Encode(input)
        self.input_len = len(tokenized_input)
        tokenized_input = [src_processing(tokenized_text, max_seq_len) for tokenized_text in [tokenized_input]]
        tokenized_input = torch.tensor(tokenized_input).to(self.device)

        # Output preallocation
        tokenized_output = torch.zeros(1, max_seq_len, dtype=torch.long).to(self.device)
        tokenized_output[0,0] = 1

        for i in range(1, max_seq_len):
            # Getting Masks Ready
            e_mask = (tokenized_input != 0).unsqueeze(1).to(self.device) # (1, 1, t)
            d_mask = (tokenized_output != 0).unsqueeze(1).to(self.device) # (1, 1, t)
            np_mask = torch.ones([1, max_seq_len, max_seq_len], dtype=torch.bool).to(self.device) # (1, t, t)
            np_mask = torch.tril(np_mask) # (1, t, t) to triangular shape
            d_mask = d_mask & np_mask # (1, t, t)

            with torch.no_grad():
                out = self.model(tokenized_input, tokenized_output, e_mask, d_mask)
                out = torch.argmax(out, dim=-1)
                last_word_id = out[0][i-1].item()

                if last_word_id == 2:
                    break

                tokenized_output[0][i] = last_word_id
        
        decoded_output = self.modern.Decode(tokenized_output[0].cpu().tolist())
        self.result = self.modern.Encode(decoded_output, out_type=str)
        self.output_len = len(self.result)
        print(decoded_output)