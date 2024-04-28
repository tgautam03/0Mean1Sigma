import torch
# Defining Embedding Layer
token_emb = torch.nn.Embedding(vocab_len, emb)
# Calling Embedding Layer
src_in = token_emb("My queen! my mistress!  O lady, weep no more")