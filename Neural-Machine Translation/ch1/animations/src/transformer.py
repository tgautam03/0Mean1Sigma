import torch
import torch.nn as nn

from src.attention import MultiHeadSelfAttention, MultiHeadCrossAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        # Normalization Layer 
        self.norm1 = torch.nn.LayerNorm([emb])
        
        # MultiHead Attention
        self.self_attention = MultiHeadSelfAttention(emb, heads)

        # Dropout Layer
        self.dropout1 = torch.nn.Dropout(0.1)

        # Normalization Layer
        self.norm2 = torch.nn.LayerNorm([emb])

        # FCN
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(emb, 4*emb),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(4*emb, emb)
        )

        # Dropout Layer
        self.dropout2 = torch.nn.Dropout(0.1)

        # Normalization Layer
        self.norm3 = torch.nn.LayerNorm([emb])

    def forward(self, x, e_mask):
        x_1 = self.norm1(x) # LayerNorm
        x_1 = self.self_attention(x_1, e_mask) # Attention 
        x_1 = self.dropout1(x_1) # Dropout
        x = x + x_1 # Skip Connection
        x_2 = self.norm2(x) # LayerNorm
        x_2 = self.fcn(x_2) # FCN
        x_2 = self.dropout2(x_2) # Dropout
        x = x + x_2 # Skip Connection
        x = self.norm3(x)
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        # Normalization Layer 
        self.norm1 = torch.nn.LayerNorm([emb])
        
        # Masked MultiHead Attention
        self.masked_self_attention = MultiHeadSelfAttention(emb, heads)

        # Dropout Layer
        self.dropout1 = torch.nn.Dropout(0.1)

        # Normalization Layer
        self.norm2 = torch.nn.LayerNorm([emb])

        # MultiHead Attention
        self.cross_attention = MultiHeadCrossAttention(emb, heads)

        # Dropout Layer
        self.dropout2 = torch.nn.Dropout(0.1)

        # Normalization Layer 
        self.norm3 = torch.nn.LayerNorm([emb])

        # FCN
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(emb, 4*emb),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(4*emb, emb)
        )

        # Dropout Layer
        self.dropout3 = torch.nn.Dropout(0.1)

        # Normalization Layer 
        self.norm4 = torch.nn.LayerNorm([emb])

    def forward(self, x, e_out, e_mask, d_mask):
        x_1 = self.norm1(x) # LayerNorm
        x_1 = self.masked_self_attention(x_1, d_mask) # Masked Attention 
        x_1 = self.dropout1(x_1) # Dropout
        x = x + x_1 # Skip Connection
        x_2 = self.norm2(x) # LayerNorm
        x_2 = self.cross_attention(x_2, e_out, e_mask) # Cross Attention 
        x_2 = self.dropout1(x_2) # Dropout
        x = x + x_2 # Skip Connection
        x_3 = self.norm3(x) # LayerNorm
        x_3 = self.fcn(x_3) # FCN
        x_3 = self.dropout2(x_3) # Dropout
        x = x + x_3 # Skip Connection
        x = self.norm4(x) # LayerNorm
        return x
    
class Transformer(torch.nn.Module):
    def __init__(self, emb, heads, max_seq_len, src_vocab_len, trg_vocab_len, num_layers=6, device="cpu"):
        super().__init__()
        # Device and layers
        self.num_layers = num_layers
        self.device = device
        
        # Token Embeddings
        self.src_token_emb = torch.nn.Embedding(src_vocab_len, emb)
        self.trg_token_emb = torch.nn.Embedding(trg_vocab_len, emb)

        # Positional Embeddings
        self.src_pos_emb = torch.nn.Embedding(embedding_dim=emb, num_embeddings=max_seq_len)
        self.trg_pos_emb = torch.nn.Embedding(embedding_dim=emb, num_embeddings=max_seq_len)

        # Encoder
        self.encoder = torch.nn.ModuleList([EncoderLayer(emb, heads) for _ in range(num_layers) ])

        # Decoder
        self.decoder = torch.nn.ModuleList([DecoderLayer(emb, heads) for _ in range(num_layers)]) 

        # Generator
        self.generator = torch.nn.Linear(emb, trg_vocab_len)

    def forward(self, src_in, trg_in, e_mask=None, d_mask=None):
        # SRC Tokenized and pos embedded
        src_in = self.src_token_emb(src_in) 
        b, t, emb = src_in.shape
        src_in = src_in + self.src_pos_emb(torch.arange(t, device=self.device))[None, :, :].expand(b, t, emb)
        
        # TRG Tokenized and pos embedded
        trg_in = self.trg_token_emb(trg_in) 
        b, t, emb = trg_in.shape
        trg_in = trg_in + self.trg_pos_emb(torch.arange(t, device=self.device))[None, :, :].expand(b, t, emb)
        
        # Encoder
        for i in range(self.num_layers):
            src_in = self.encoder[i](src_in, e_mask)
        
        # Decoder
        for i in range(self.num_layers):
            trg_in = self.decoder[i](trg_in, src_in, e_mask, d_mask)
        
        # Generator
        out = torch.nn.LogSoftmax(dim=-1)(self.generator(trg_in))
        

        return out