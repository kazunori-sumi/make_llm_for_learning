import urllib.request
import re
import tiktoken
import torch
from self_attention import SelfAttention_v2
from causal_attention import CausalAttention
from multi_head_attention_wrapper import MultiHeadAttentionWrapper
from multi_head_attention import MultiHeadAttention
from importlib.metadata import version
from dataloader import GPTDatasetV1
from torch.utils.data import DataLoader

def retrieve_raw_text():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt"
           )
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def parse_token(raw):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token:integer for integer,token in enumerate(all_words)}
    return vocab

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers # 前処理用のCPUプロセス数
    )

    return dataloader

def create_embeddings(inputs, vocab_size, context_length, output_dim):
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    return token_embeddings, pos_embeddings

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

def softmax_stable(x):
    return torch.exp(x - x.max()) / torch.exp(x - x.max()).sum()

def main():
    # with open("the-verdict.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()

    # vocab_size = 50257
    # output_dim = 256
    # max_length = 4

    # dataloader = create_dataloader_v1(
    #     raw_text,
    #     batch_size=8,
    #     max_length=max_length,
    #     stride=max_length,
    #     shuffle=False
    # )
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print("Token ids:\n", inputs)
    # print("\nInputs: shape\n", inputs.shape)

    # token_embeddings, pos_embeddings = create_embeddings(inputs, vocab_size, max_length, output_dim)
    # print(token_embeddings.shape)
    # print(pos_embeddings.shape)

    # input_embeddings = token_embeddings + pos_embeddings
    # print(input_embeddings.shape)


    attn_inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )

    x_2 = attn_inputs[1]
    d_in = attn_inputs.shape[1]
    d_out = 2
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(attn_inputs))

    queries = sa_v2.W_query(attn_inputs)
    keys = sa_v2.W_key(attn_inputs)

    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    # print(attn_weights)
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    # print(mask_simple)
    masked_simple = attn_weights * mask_simple
    # print(masked_simple)
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    # print(masked_simple_norm)

    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    # print(masked)
    attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=1)
    # print(attn_weights)

    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    # print(dropout(example))
    # print(dropout(attn_weights))

    batch = torch.stack((attn_inputs, attn_inputs), dim=0)
    # print(batch.shape)

    # torch.manual_seed(123)
    # context_length = batch.shape[1]
    # ca = CausalAttention(d_in, d_out, context_length, 0.0)
    # context_vecs = ca(batch)
    # print(context_vecs.shape)

    # torch.manual_seed(123)
    # context_length = batch.shape[1]
    # d_in, d_out = 3,2
    # mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    # context_vecs = mha(batch)

    # print(context_vecs)
    # print(context_vecs.shape)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print(context_vecs.shape)

if __name__ == "__main__":
    main()