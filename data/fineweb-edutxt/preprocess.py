import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# load fineweb-edu with parallel processing
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64, cache_dir="/your/cache/path")

"""
Check for available data configuration: like HuggingFace/fineweb-edu, 1b token or 10b tokenor 100b token etc.
"""
from datasets import get_dataset_config_names
print(get_dataset_config_names("HuggingFace/fineweb-edu"))

"""
Download dataset of specific configuration on your cache directory.
"""
# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                       name="sample-100BT", 
                       num_proc=64, 
                       cache_dir="/data/huggingface_cache")  # set cache directory properly
                       # local directory: "c:\\Users\\username\\huggingface_datasets"
                       # or "D:\\dataset\\huggingface_datasets"

"""
Or, Setting a Global cache directory: You can tell Hugging Face to always 
use your preferred directory for caching datasets and models.

import os

os.environ["HF_DATASETS_CACHE"] = "D:\\Anima\\dataset\\huggingface_dataset"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Anima\\model\\hugingface_model"

# download data, model, and tokenizer

from datasets import load_dataset
dataset = load_dataset("HuggingFace_fineweb-edu",
                        name = "sample-10BT",
                        num_proc = 8)

from transformer import AutoModel, AutoTokenizer
model = Automodel.pre_trained("model_name)
tokenizer = AutoTokenizer.pre_trained("model_name)
"""


"""
Inspect dataset...
"""
print(dataset)
print(dataset["train"][0])   # This be like:---> {"text": "hello, this is the techno world in India"}
print(dataset["train"].select(range(10000)))  # --> It will give 10k samples for testing.

"""
Streaming mode is recommended for large dataset, but keep in mind your internet connection 
should be very good so it won't cause streaming error which eventually cause training issue.
"""
# dataset = load_dataset("HuggingFace/fineweb-edu",
#                        name = "sample-100BT",
#                        num_proc = 64,
#                        streaming = True)

# for example in dataset["train"]:
#     print(example["text"])
#     break

"""
TOKENIZATION IS THE MOST IMPORTENT PART IN MODEL TRAINING.

THERE ARE TWO WAY: 
1.) WHETHER USE PRE-BUILT TOKENIZER LIKE:
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

or FROM HggingFaceHub
from transformers import AutoTokenizer
tokenizer = Autotokenizer.pre_trained("gpt2")
tokenizer.save_pretrained("path")


2.) OR, BUILD YOUR OWN USING ALGORITHMS LIKE BPE, SENTENCEPIECE
import tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(["data_path"], vocab_size = 32000, min_frequency=2)
tokenizer.save_model("/path/to/save/tokenizer")
"""
# Taking gpt2 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.pre_trained("gpt2")

# GPT-2 does not have a pad token → fix that
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
           example["text"],
           truncation = True,
           max_length = 1024,
           padding = "max_length",
           )

tokenized_data = dataset.map(
                        tokenize_function,
                        batched = True,
                        num_proc = 8,   # parallel processing.
                        remove_columns = ["text"]  # removes raw text, keeps only tokenized form
                        )

# check it...
print(tokenized_data[0])

"""
Group the tokens for training of chunks of fixed length /// Or DataLoader
"""
# block_size = 1024

# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated["input_ids"])
#     # Drop remainder tokens.
#     total_length = (total_length // block_size) * block_size
#     # Split into chunks of block_size.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# lm_dataset = tokenized_data.map(
#     group_texts,
#     batched=True,
#     num_proc=8
# )
"""
Now it is ready to go for training, but before giving lm_dataset to the model must use 
COLLATE-function for dynamic padding, batching, masking, tensor conversion and labeling.

USED in training:

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)
"""
"""
Save groupd data
"""
# lm_dataset.save_to_disk("path")

# # Load later:
# from datasets import load_from_disk
# lm_dataset = load_from_disk("path")