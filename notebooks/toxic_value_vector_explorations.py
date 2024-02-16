# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns

# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

pio.renderers.default = "png"

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

import plotly.graph_objects as go

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML


import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
torch.set_grad_enabled(False)

## Helper and Visual Functions
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

## Load the Models GPT2-Small, SOLU 1-4
CUDA_LAUNCH_BLOCKING=1
model = HookedTransformer.from_pretrained(
    "gpt-j-6B",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# Getting the FV
FV = np.load("Antonym_FV.npy", allow_pickle=True)

#This is how you would run the model on a prompt
example_prompt = "up:down, left:right, increase:"

#Control prompts test
example_answer = "decrease"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_token = model.to_tokens("decrease", prepend_bos=False)
tokens = model.to_tokens(example_prompt)
print(example_prompt)
print(answer_token)

"""# Initial Explorations"""

# Move the tokens to the GPU
tokens = tokens.cuda()

# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)

# Logit Lens

# Returns the accumulated residual stream at each layer/sub-layer and apply_ln applies final layer norm
accumulated_residual, labels = cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, apply_ln=True, return_labels=True
)

print(accumulated_residual.shape)

#Project each layer residual at last token onto vocab space
projection_vocab = einsum("layer d_model, d_model d_vocab --> layer d_vocab", accumulated_residual[:,-1,:], model.W_U)

#Look at the value for each layer at correct token index
line(
    projection_vocab[:,answer_token],
    hover_name=labels,
    title="Value of Correct Answer Logit",
)

# Look at each component
decomposed_residual, labels = cache.decompose_resid(
    layer=-1, apply_ln=True, return_labels=True
)

# Project each layer and each position onto vocab space
projection_vocab = einsum("layer pos d_model, d_model d_vocab --> layer pos d_vocab", decomposed_residual[:,-1,:,:], model.W_U)

print(projection_vocab.shape)
probs = projection_vocab.softmax(dim=-1)

#Look at the value for each layer at ' shit' index
line(
    projection_vocab[:,-1,answer_token],
    hover_name=labels,
    title="Value of Answer Logit at each component",
)

len(labels)

decomposed_residual, labels = cache.decompose_resid(
    layer=-1, apply_ln=True, return_labels=True
)
