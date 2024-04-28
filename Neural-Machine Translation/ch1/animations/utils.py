import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from manim import *

def NN(num_input_neurons, hidden, num_output_neurons):
    input_layer = [Circle(color=BLUE)]
    prev_neuron = input_layer[0]
    for neuron in range(1, num_input_neurons) :
        layer_neuron = Circle(color=BLUE).next_to(prev_neuron, DOWN).shift(0.5*DOWN)
        input_layer.append(layer_neuron)
        prev_neuron = layer_neuron
    input_box = SurroundingRectangle(VGroup(*input_layer), color=WHITE)

    hidden_layers = []
    hidden_boxes = []
    prev_layer = input_box
    for num_hidden_neurons in hidden:
        hidden_layer = [Circle(color=BLUE)]
        prev_neuron = hidden_layer[0]
        for neuron in range(1, num_hidden_neurons) :
            layer_neuron = Circle(color=BLUE).next_to(prev_neuron, DOWN).shift(0.5*DOWN)
            hidden_layer.append(layer_neuron)
            prev_neuron = layer_neuron
        VGroup(*hidden_layer).next_to(prev_layer, RIGHT).shift(3*RIGHT)
        hidden_box = SurroundingRectangle(VGroup(*hidden_layer), color=WHITE)
        hidden_layers.append(VGroup(*hidden_layer))
        hidden_boxes.append(hidden_box)
        prev_layer = hidden_box

    output_layer = [Circle(color=BLUE)]
    prev_neuron = output_layer[0]
    for neuron in range(1, num_output_neurons) :
        layer_neuron = Circle(color=BLUE).next_to(prev_neuron, DOWN).shift(0.5*DOWN)
        input_layer.append(layer_neuron)
        prev_neuron = layer_neuron
    VGroup(*output_layer).next_to(prev_layer, RIGHT).shift(3*RIGHT)
    output_box = SurroundingRectangle(VGroup(*output_layer), color=WHITE)

    
    connections = []
    layer_1 = input_layer
    layer_2 = hidden_layers[0]
    for neuron_1 in layer_1:
        for neuron_2 in layer_2:
            connection = Line(neuron_1.get_edge_center(RIGHT), neuron_2.get_edge_center(LEFT), color=GREEN).set_stroke(width=2.5)
            connections.append(connection)

    for i in range(1, len(hidden_layers)):
        layer_1 = hidden_layers[i-1]
        layer_2 = hidden_layers[i]
        for neuron_1 in layer_1:
            for neuron_2 in layer_2:
                connection = Line(neuron_1.get_edge_center(RIGHT), neuron_2.get_edge_center(LEFT), color=GREEN).set_stroke(width=2.5)
                connections.append(connection)
    layer_1 = hidden_layers[-1]
    layer_2 = output_layer
    for neuron_1 in layer_1:
        for neuron_2 in layer_2:
            connection = Line(neuron_1.get_edge_center(RIGHT), neuron_2.get_edge_center(LEFT), color=GREEN).set_stroke(width=2.5)
            connections.append(connection)

    return input_layer, input_box, hidden_layers, hidden_boxes, output_layer, output_box, connections

def remove_invisible_chars(mobject: SVGMobject) -> SVGMobject:
    """Function to remove unwanted invisible characters from some mobjects.

    Parameters
    ----------
    mobject
        Any SVGMobject from which we want to remove unwanted invisible characters.

    Returns
    -------
    :class:`~.SVGMobject`
        The SVGMobject without unwanted invisible characters.
    """
    # TODO: Refactor needed
    iscode = False
    if mobject.__class__.__name__ == "Text":
        mobject = mobject[:]
    elif mobject.__class__.__name__ == "Code":
        iscode = True
        code = mobject
        mobject = mobject.code
    mobject_without_dots = VGroup()
    if mobject[0].__class__ == VGroup:
        for i in range(len(mobject)):
            mobject_without_dots.add(VGroup())
            mobject_without_dots[i].add(*(k for k in mobject[i] if k.__class__ != Dot))
    else:
        mobject_without_dots.add(*(k for k in mobject if k.__class__ != Dot))
    if iscode:
        code.code = mobject_without_dots
        return code
    return mobject_without_dots

##############################################################
###########Token Embedding and Positional Encoding############
##############################################################
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

def animate_emb(embs, tokens, actual_token_nums=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    data = embs[0].detach().numpy()[0][:actual_token_nums,:16].T
    im = ax.matshow(data)
    # Create text objects to display the values
    text_objects = []
    for (i, j), z in np.ndenumerate(data):
        text = ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'), color="black")
        text_objects.append(text)
    ax.set_xticks(range(len(tokens[:actual_token_nums])), tokens[:actual_token_nums], size='small')

    # Define the animation function
    def animate(i):
        # Update the data
        data = embs[i].detach().numpy()[0][:actual_token_nums,:16].T
        im.set_data(data)

        if i == 0 or i == len(embs)-1:
            plt.pause(5)

        # Update the text objects
        for j, text in enumerate(text_objects):
            row = j // data.shape[1]
            col = j % data.shape[1]
            text.set_text(f"{data[row, col]:.3f}")

        return [im] + text_objects

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(embs), interval=1, blit=True)

    return ani

##############################################################
########################Encoder###############################
##############################################################
def show_emb(embs, tokens, actual_token_nums=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    data = embs[-1].detach().numpy()[0][:actual_token_nums,:16].T
    im = ax.matshow(data)
    # Create text objects to display the values
    text_objects = []
    for (i, j), z in np.ndenumerate(data):
        text = ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'), color="black")
        text_objects.append(text)
    ax.set_xticks(range(len(tokens[:actual_token_nums])), tokens[:actual_token_nums], size='small')

    return fig

def animate_attn(embs, tokens, actual_token_nums=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    data = embs[0].detach().numpy()[0][:actual_token_nums,:actual_token_nums]
    im = ax.matshow(data)
    # Create text objects to display the values
    text_objects = []
    for (i, j), z in np.ndenumerate(data):
        text = ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'), color="black")
        text_objects.append(text)
    ax.set_xticks(range(len(tokens[:actual_token_nums])), tokens[:actual_token_nums], size='small')
    ax.set_yticks(range(len(tokens[:actual_token_nums])), tokens[:actual_token_nums], size='small')

    # Define the animation function
    def animate(i):
        # Update the data
        data = embs[i].detach().numpy()[0][:actual_token_nums,:actual_token_nums]
        im.set_data(data)

        if i == 0 or i == len(embs)-1:
            plt.pause(5)

        # Update the text objects
        for j, text in enumerate(text_objects):
            row = j // data.shape[1]
            col = j % data.shape[1]
            text.set_text(f"{data[row, col]:0.3f}")

        return [im] + text_objects

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(embs), interval=1, blit=True)

    return ani