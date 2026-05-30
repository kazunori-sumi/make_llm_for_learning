import urllib.request
import re
import tiktoken
import torch

from dummy_gpt_model import DummyGPTModel

def main():
    GPT_CONFIG_124M = {
        "vocab_size" : 50257,
        "context_length" : 1024,
        "emb_dim" : 768,
        "n_heads" : 12,
        "n_layers" : 12,
        "drop_rate" : 0.1,
        "qkv_bias" : False
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1="Every effort moves you"
    txt2="Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    
    
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits)

if __name__ == "__main__":
    main()