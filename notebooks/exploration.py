import os, re, json
import torch, numpy as np
import sys

sys.path.append('..')
torch.set_grad_enabled(False)

# %%
import matplotlib.pyplot as plt
from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from src.utils.intervention_utils import fv_intervention_natural_text, function_vector_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.eval_utils import decode_to_vocab, sentence_eval


model_name = 'EleutherAI/gpt-j-6b'
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

dataset = load_dataset('country-capital', seed=0)

mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)

FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)
# %%

dataset = load_dataset('country-capital')
word_pairs = dataset['train'][:5]
test_pairs = dataset['test'][:5]

zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pairs, prepend_bos_token=True, shuffle_labels=True)
zeroshot_sentence = create_prompt(zeroshot_prompt_data)
print("Zero-Shot Prompt:\n", repr(zeroshot_sentence))
#breakpoint()
total_diffs = []
for i in range(len(test_pairs['input'])):
    diffs = []
    for EDIT_LAYER in [x for x in range(0,25)]:
        test_pair = [test_pairs['input'][i], test_pairs['output'][i]]
        # Intervention on the zero-shot prompt
        clean_logits, interv_logits = function_vector_intervention(zeroshot_sentence, [test_pair[1]], EDIT_LAYER, FV, model, model_config, tokenizer)
       # breakpoint()
        index = tokenizer(test_pair[1])['input_ids'][0]
        diffs.append((interv_logits[0][index]-clean_logits[0][index]).cpu().item())
    plt.plot(diffs)
    total_diffs.append(diffs)
breakpoint()
