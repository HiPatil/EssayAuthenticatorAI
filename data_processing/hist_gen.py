import sys


import sys
import fairscale
import os
import torch
import pandas as pd
import random
import string
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.generation import Llama, Dialog
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

sys.path.append('/projectnb/textconv/llama/packages')

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_cuda_devices}")

    # List the properties of each CUDA device
    for i in range(num_cuda_devices):
        device = torch.device(f'cuda:{i}')
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available on this system.")
    
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888' #since i am doing my llama stuff already haha
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

generator = Llama.build(
        ckpt_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        max_seq_len=512, #max_seq_len....
        max_batch_size=6,
    )

def tokenize_sentence(sentence):
    prompt_tokens = generator.tokenizer.encode(
                        sentence,
                        bos=True,
                        eos=True,
    )
    return torch.tensor([prompt_tokens])

def selection_tensor(model, indices, vocab_dim = 32_000):
    """
    model will be llama, give it generator.model, and indices should be a tensor of batch 1
    so will look like tensor([[    1,   910,  3686,   388}]])
    not sure we can vectorize this?
    """
    rank_list = []
    seq_len = len(indices[0])
    
    #i think these are right, all the indexing.  allow the prints to prove it.
    
    for i in range(1,seq_len): #maybe we skip the first one? idk ask the boyz
        
        #print(indices[:,0:i],indices[:,i])
        
        model_result = model.forward(indices[:,0:i],0) 
        data_tensor = model_result[:, -1]  #temperature here if you wana
        sorted_data, _ = data_tensor.sort(dim=1, descending=True)
        indices_tensor = indices[:,i].unsqueeze(0) #get the next token, to compare to the model result
        ranks = torch.where(sorted_data == data_tensor.gather(1, indices_tensor), torch.arange(1, vocab_dim + 1).unsqueeze(0), torch.zeros(1))
        ranks = int(torch.max(ranks,axis =-1).values)
        rank_list.append(ranks)
    return rank_list

def large_selection_tensor(model, indices, vocab_dim = 32_000, max_seq_len=512):
    """
    model will be llama, give it generator.model, and indices should be a tensor of batch 1
    so will look like tensor([[    1,   910,  3686,   388}]])
    not sure we can vectorize this?
    """
    rank_list = []
    seq_len = len(indices[0])
    
    #i think these are right, all the indexing.  allow the prints to prove it.
    
    for i in range(1,seq_len): #maybe we skip the first one? idk ask the boyz
        
        #print(indices[:,0:i],indices[:,i])
        
        start_idx = max(0,i-max_seq_len)
        
        model_result = model.forward(indices[:,start_idx:i],0) 
        #print(start_idx, model_result.shape)
        data_tensor = model_result[:, -1]  #temperature here if you wana
        sorted_data, _ = data_tensor.sort(dim=1, descending=True)
        indices_tensor = indices[:,i].unsqueeze(0) #get the next token, to compare to the model result
        ranks = torch.where(sorted_data == data_tensor.gather(1, indices_tensor), torch.arange(1, vocab_dim + 1).unsqueeze(0), torch.zeros(1))
        ranks = int(torch.max(ranks,axis =-1).values)
        rank_list.append(ranks)
    return rank_list

def process_essay(essay_text, model):
    essay_tokens = tokenize_sentence(essay_text)
    idx_list = large_selection_tensor(model, essay_tokens)
    return idx_list

def process_essays(model, input_df, result_file):
    try:
        # Load the existing result file if it exists
        new_df = pd.read_csv(result_file)
        processed_ids = set(new_df["id"].tolist())
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame and dictionary
        new_df = pd.DataFrame(columns=["id", "indexes"])
        processed_ids = {}

    # Iterate through the original DataFrame
    for index, row in input_df.iterrows():
        essay_id = row["id"]
        essay_text = row["text"]

        # Check if the ID has already been processed
        if essay_id in processed_ids:
            print(f"ID {essay_id} has already been processed. Skipping...")
            continue

        try:
            # Process the essay text
            indexes = process_essay(essay_text, model)
            max_idx = new_df.index.max()
            write_idx = max_idx + 1
            new_df.loc[write_idx] = [str(essay_id),indexes]
            new_df.to_csv(result_file, index=False)
            processed_ids.add(essay_id)
            
            print(f"Processed ID {essay_id} successfully.")

        except Exception as e:
            print(f"Error processing ID {essay_id}: {str(e)}")

    # After processing all essays, save the final new DataFrame
    new_df.to_csv(result_file, index=False)
    
df = pd.read_csv('train_essays_combined.csv')
process_essays(generator.model, input_df = df, result_file = "llama_hist.csv")