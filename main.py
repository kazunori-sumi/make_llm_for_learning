import urllib.request
import re
import tiktoken
import torch
from simple_tokenizer import SimpleTokenizerV2
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

def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    max_length = 4

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token ids:\n", inputs)
    print("\nInputs: shape\n", inputs.shape)

    token_embeddings, pos_embeddings = create_embeddings(inputs, vocab_size, max_length, output_dim)
    print(token_embeddings.shape)
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

if __name__ == "__main__":
    main()