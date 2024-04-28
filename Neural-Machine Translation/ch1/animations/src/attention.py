import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi Head Self Attention
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        assert emb % heads == 0
        self.emb, self.heads = emb, heads

        # Three emb x emb matrix multiplications to get queries, keys and values
        self.to_queries = torch.nn.Linear(emb, emb)
        self.to_keys = torch.nn.Linear(emb, emb)
        self.to_values = torch.nn.Linear(emb, emb)

        # One last Linear layer at the end with emb x emb matrix multiplication
        self.unify = torch.nn.Linear(emb, emb)

    def masked_attention(self, queries, keys, values, mask=None):
        W = torch.matmul(queries, keys.transpose(-2,-1)) # Computing Weights
        W = W / (self.emb**(1/2)) # Scaling for stability
        # Applying Mask
        if mask is not None:
            mask = mask.unsqueeze(1) # (b, 1, t) => (b, 1, 1, t) or (b, t, t) => (b, 1, t, t)
            W = W.masked_fill_(mask == 0, -1e+10)
        W = F.softmax(W, dim=-1) # Row-wise Softmax
        y = torch.matmul(W, values) # Computing y

        self.attention_weights = W

        return y

    def forward(self, x, mask=None):
        b, t, emb = x.shape # Batch Size, Sequence Length, embedding dim
        h = self.heads
        k = emb//h

        # Computing queries, keys and values
        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        # Slicing out the heads
        queries = queries.view(b, t, h, k)
        keys = keys.view(b, t, h, k)
        values = values.view(b, t, h, k)

        # Putting heads next to batch dims (Remember head computations can run in parallel)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Here comes Attention...
        y = self.masked_attention(queries, keys, values, mask)
        
        # Concat heads
        y = y.transpose(1, 2).reshape(b, t, emb) # Concatenating heads

        # Final Linear NN Layer
        return self.unify(y)
    
# Multihead Cross Attention
class MultiHeadCrossAttention(torch.nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        assert emb % heads == 0
        self.emb, self.heads = emb, heads

        # Three emb x emb matrix multiplications to get queries, keys and values
        self.to_queries = torch.nn.Linear(emb, emb)
        self.to_keys = torch.nn.Linear(emb, emb)
        self.to_values = torch.nn.Linear(emb, emb)

        # One last Linear layer at the end with emb x emb matrix multiplication
        self.unify = torch.nn.Linear(emb, emb)

    def masked_attention(self, queries, keys, values, mask=None):
        W = torch.matmul(queries, keys.transpose(-2,-1)) # Computing Weights
        W = W / (self.emb**(1/2)) # Scaling for stability
        # Applying Mask
        if mask is not None:
            mask = mask.unsqueeze(1) # (b, 1, t) => (b, 1, 1, t) or (b, t, t) => (b, 1, t, t)
            W = W.masked_fill_(mask == 0, -1e+10)
        W = F.softmax(W, dim=-1) # Row-wise Softmax
        y = torch.matmul(W, values) # Computing y

        self.attention_weights = W

        return y

    def forward(self, x1, x2, mask=None):
        b, t, emb = x1.shape # Batch Size, Sequence Length, embedding dim
        h = self.heads
        k = emb//h

        # Computing queries, keys and values
        queries = self.to_queries(x1)
        keys = self.to_keys(x2)
        values = self.to_values(x2)

        # Slicing out the heads
        queries = queries.view(b, t, h, k)
        keys = keys.view(b, t, h, k)
        values = values.view(b, t, h, k)

        # Putting heads next to batch dims (Remember head computations can run in parallel)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Here comes Attention...
        y = self.masked_attention(queries, keys, values, mask)
        
        # Concat heads
        y = y.transpose(1, 2).reshape(b, t, emb) # Concatenating heads

        # Final Linear NN Layer
        return self.unify(y)