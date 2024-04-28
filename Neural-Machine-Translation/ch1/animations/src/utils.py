import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def to_tokens(stn, tokenizer, max_seq_len, stn_type):
    # Tokenized stn
    if stn_type == "src" or stn_type == "trg_out":
        stn_tok = tokenizer.EncodeAsPieces(list(stn)[0])
        stn_tok += ["[EOS]"] # EOS Token string
        stn_tok = (stn_tok + max_seq_len * ['[PAD]'])[:max_seq_len] # Padding Tokens string
        stn_tok = [piece.lstrip('▁') for piece in stn_tok]
    elif stn_type == "trg_in":
        stn_tok = tokenizer.EncodeAsPieces(list(stn)[0])
        stn_tok = ["[BOS]"] + stn_tok # BOS Token string
        stn_tok = (stn_tok + max_seq_len * ['[PAD]'])[:max_seq_len] # Padding Tokens string
        stn_tok = [piece.lstrip('▁') for piece in stn_tok]
    else:
        Exception("str_type: {src, trg_in, trg_out}")

    return stn_tok

def animate_emb(embs, tokens):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    data = embs[0].detach().numpy()[0]
    im = ax.matshow(data)
    # Create text objects to display the values
    text_objects = []
    for (i, j), z in np.ndenumerate(data):
        text = ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'), color="black")
        text_objects.append(text)
    ax.set_yticks(range(len(tokens)), tokens, size='small')
    ax.set_ylabel("Tokenized Sentence")
    ax.set_xlabel("Embeddings")

    # Define the animation function
    def animate(i):
        # Update the data
        data = embs[i].detach().numpy()[0]
        im.set_data(data)

        # Update the text objects
        for j, text in enumerate(text_objects):
            row = j // data.shape[1]
            col = j % data.shape[1]
            text.set_text(f"{data[row, col]:.1f}")

        return [im] + text_objects

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(embs), interval=1, blit=True)

    return ani