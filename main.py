import urllib.request
import re
import tiktoken
import torch
import torch.nn as nn

from gpt_model import GPTModel
from gpt_model import LayerNorm
from feed_forward import FeedForward
from feed_forward import ExampleDeepNuralNetwork
from transformer_block import TransformerBlock

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
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print(batch)
    print(out.shape)
    print(out)

    # batch_example = torch.randn(2,5)
    # layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
    # out = layer(batch_example)
    # print(out)

    # ffn = FeedForward(GPT_CONFIG_124M)

    # x = torch.rand(2,3,768)
    # out = ffn(x)
    # print(out.shape)

    # layer_sizes = [3,3,3,3,3,1]
    # sample_input = torch.tensor([[1.,0.,-1.]])

    # torch.manual_seed(123)
    # model_without_shortcut = ExampleDeepNuralNetwork(
    #     layer_sizes, use_shortcut=False
    # )

    # print_gradients(model_without_shortcut, sample_input)

    # x = torch.rand(2,4,768)
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)

    # print(x.shape)
    # print(output.shape)

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param, in model.named_parameters():
        if 'weight' in name:
            print(
                f"{name} has gradient mean of "
                f"{param.grad.abs().mean().item()}"
            )

if __name__ == "__main__":
    main()