import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import argparse

from processing import src_processing, input_trg_processing, output_trg_processing
from transformer import Transformer

if __name__ == "__main__":
    # Getting variables
    parser = argparse.ArgumentParser(description='Training Transformer')
    parser.add_argument('--emb', type=int, required=True, help='Embedding Dimension')
    parser.add_argument('--heads', type=int, required=True, help='Number of Attention Heads')
    parser.add_argument('--layers', type=int, required=True, help='Number of encoder/decoder layers')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Minibatch Size')

    args = parser.parse_args()

    # Change Working Directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ####################################
    ######### Loading Dataset ##########
    ####################################
    df = pd.read_csv("../data/shk2mod.csv", index_col=0) # Loading the csv file
    df.drop("id", axis=1, inplace=True)
    d = df.to_numpy()           # Converting dataframe to numpy
    src, trg = d[:,0], d[:,1]   # Splitting columns into source and target
    print("SRC shape: {}; TRG shape: {}".format(src.shape, trg.shape))

    ####################################
    ######## Tokenizing Sentences ######
    ####################################
    # Encoding src
    shakespeare = spm.SentencePieceProcessor(model_file="../trained_models/tokenizer/shakespeare_en.model")
    src_id = shakespeare.EncodeAsIds(list(src))
    # Encoding trg
    modern = spm.SentencePieceProcessor(model_file="../trained_models/tokenizer/modern_en.model")
    trg_id = modern.EncodeAsIds(list(trg))

    ####################################
    ##### Padding and Preprocessing ####
    ####################################
    # Max Sequence Length
    max_seq_len = 256
    # Padding/Truncating src
    src_id = [src_processing(id, max_seq_len) for id in src_id] 
    # Padding/Truncating trg
    input_trg_id = [input_trg_processing(id, max_seq_len) for id in trg_id] 
    output_trg_id = [output_trg_processing(id, max_seq_len) for id in trg_id] 
    # Moving everything to torch tensors
    src_id = torch.tensor(src_id)
    input_trg_id = torch.tensor(input_trg_id)
    output_trg_id = torch.tensor(output_trg_id)
    print("Src Shapes: {}; dtype: {}".format(src_id.shape, src_id.dtype))
    print("Input Trg Shapes: {}; dtype: {}".format(input_trg_id.shape, input_trg_id.dtype))
    print("Output Trg Shapes: {}; dtype: {}".format(output_trg_id.shape, output_trg_id.dtype))

    ####################################
    ############# Training #############
    ####################################
    # Batch Size
    batch_size = 180
    # PyTorch DataLoaders
    batch_size = args.batch_size
    src_loader = torch.utils.data.DataLoader(dataset=src_id, batch_size=batch_size)
    input_trg_loader = torch.utils.data.DataLoader(dataset=input_trg_id, batch_size=batch_size)
    output_trg_loader = torch.utils.data.DataLoader(dataset=output_trg_id, batch_size=batch_size)
    # Hyperparameters
    lr = 0.001
    num_epochs = 300
    device = "cuda"
    emb, heads, layers = args.emb, args.heads, args.layers
    print("Embedding Dim: {}; Attention Heads: {}, Number of layers: {}; Batch Size: {}".format(emb, heads, layers, batch_size))
    model = Transformer(emb=emb, heads=heads, max_seq_len=max_seq_len, 
                        src_vocab_len=shakespeare.vocab_size(), trg_vocab_len=modern.vocab_size(),
                        num_layers=layers, device=device).to(device)
    model.load_state_dict(torch.load("../trained_models/model_emb256_heads16_layers5.pt"))
    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    # train_loss_record = []
    train_loss_record = list(np.load("../trained_models/train_loss_record.npy"))

    start_err = train_loss_record[-1]
    print("Start error: ", start_err)
    print("Training...")
    for epoch in range(num_epochs+1):
        train_losses = []
        for (src_in, trg_in, trg_out) in zip(src_loader, input_trg_loader, output_trg_loader):
            # Clear gradients
            opt.zero_grad()

            # Move data to GPU
            src_in = src_in.to(device)
            trg_in = trg_in.to(device)
            trg_out = trg_out.to(device)

            # Compute Masks
            e_mask = (src_in != 0).unsqueeze(1).to(device) # (b, 1, t)
            d_mask = (trg_in != 0).unsqueeze(1).to(device) # (b, 1, t)
            np_mask = torch.ones([1, max_seq_len, max_seq_len], dtype=torch.bool).to(device) # (1, t, t)
            np_mask = torch.tril(np_mask) # (1, t, t) to triangular shape
            d_mask = d_mask & np_mask # (b, t, t)
            
            # Getting Output
            out = model(src_in, trg_in, e_mask.to(device), d_mask.to(device))

            # Loss
            loss = torch.nn.NLLLoss(ignore_index=0)(out.view(-1, modern.vocab_size()), trg_out.view(-1, ))

            # Update Weights
            loss.backward()
            opt.step()

            # Store Losses
            train_losses.append(loss.item())

            del src_in, trg_in, trg_out, e_mask, d_mask, out
            torch.cuda.empty_cache()

        avg_train_loss = np.mean(train_losses)
        print("Ep {}; Train Loss: {}".format(epoch, avg_train_loss))
        
        train_loss_record.append(avg_train_loss)
        np.save("../trained_models/train_loss_record.npy", train_loss_record)
        if epoch % 25 == 0: 
            torch.save(model.state_dict(), "../trained_models/nn_models/model_emb{}_heads{}_layers{}_ep{}.pt".format(emb, heads, layers, epoch))
        
        if avg_train_loss < start_err:
            print("Saving model at epoch {}".format(epoch))
            torch.save(model.state_dict(), "../trained_models/model_emb{}_heads{}_layers{}.pt".format(emb, heads, layers))
            start_err = avg_train_loss