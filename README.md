


## Downloading section:

"""
Setting a Global cache directory for your model, and data: You can tell Hugging Face to always 
use your preferred directory for caching datasets and models.

### On terminal before runnign python
export HF_DATASETS_CACHE = "directory"
export TRANSFORMERS_CACHE = "directory"


### Inside script
import os

os.environ["HF_DATASETS_CACHE"] = "D:\\Anima\\dataset\\huggingface_dataset"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Anima\\model\\hugingface_model"

#### download data, model, and tokenizer

#### Data
from datasets import load_dataset
dataset = load_dataset("HuggingFace_fineweb-edu",
                        name = "sample-10BT",
                        num_proc = 8)

#### Model, & Tokenizer
from transformer import AutoModel, AutoTokenizer
model = Automodel.pre_trained("model_name)
tokenizer = AutoTokenizer.pre_trained("model_name)
"""