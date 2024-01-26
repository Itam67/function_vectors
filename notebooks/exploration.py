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





breakpoint()
# %%
dataset = load_dataset('country-capital')
test_pairs = dataset['test'][:]

sims = []
for i in range(len(test_pairs['input'])):
    test_pair = [test_pairs['input'][i], test_pairs['output'][i]]
    index = tokenizer(' '+test_pair[1])['input_ids'][0]
    unembedding = model.lm_head.weight[index,:]
    sims.append(FV[0].dot(unembedding).cpu().item())
    
plt.figure()
plt.plot(sims)
plt.title('Dot Products of FV with Answer Unembeddings')
plt.xlabel('Prompt #')
plt.ylabel('Dot Product')
plt.savefig('FV_Dot.png')

breakpoint()
# total_diffs = []
accs = []
cos_sims = []

for EDIT_LAYER in range(0,25):
    accuracy = 0

    for i in range(len(test_pairs['input'])):
        # diffs = []
        test_pair = [test_pairs['input'][i], test_pairs['output'][i]]
        test_pair_dict = {'input': test_pair[0], 'output': test_pair[1]}
        zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair_dict, prepend_bos_token=True, shuffle_labels=True)
        
        zeroshot_sentence = create_prompt(zeroshot_prompt_data)

        # Intervention on the zero-shot prompt
        clean_logits, interv_logits = function_vector_intervention(zeroshot_sentence, [test_pair[1]], EDIT_LAYER, FV, model, model_config, tokenizer)
       # breakpoint()
        index = tokenizer(' '+test_pair[1])['input_ids'][0]
        # diffs.append(((interv_logits[0][index]-clean_logits[0][index])/clean_logits[0][index]).cpu().item())
        # breakpoint()
        # print(tokenizer.decode(interv_logits[0].argmax()), 
            #   tokenizer.decode(index),
            #   interv_logits[0].argmax().item()==index)
        if interv_logits[0].argmax().item()==index:
            accuracy+=1
        
    # plt.plot(diffs)
    # total_diffs.append(diffs)
    accs.append(accuracy/len(test_pairs['input']))
plt.title("Country-Capital FV Interventions")
plt.plot(accs)
plt.xlabel('Intervention Layer')
plt.ylabel('Accuracy Across All Prompts')
plt.savefig('acc.png')
plt.figure()
# plt.plot(np.array(total_diffs).mean(axis=0))
# plt.xlabel('Intervention Layer')
# plt.ylabel('% Change in logit for correct token at last layer')
# plt.savefig('averaged.png')

breakpoint()
