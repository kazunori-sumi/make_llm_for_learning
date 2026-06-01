import urllib.request
import re
import tiktoken
import torch
import torch.nn as nn

from dummy_gpt_model import DummyGPTModel
from dummy_gpt_model import LayerNorm

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

    batch_example = torch.randn(2,5)
    layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
    out = layer(batch_example)
    print(out)

if __name__ == "__main__":
    main()