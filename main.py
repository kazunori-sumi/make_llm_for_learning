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

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nInputs:\n", targets)

    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))

    # enc_text = tokenizer.encode(raw_text)
    # enc_sample = enc_text[50:]

    # context_size = 4
    # x = enc_sample[:context_size]
    # y = enc_sample[1:context_size + 1]
    # print(f"x: {x}")
    # print(f"y:      {y}")

    # # スライディングウィンドウによる訓練データのペアを作成
    # # このコンテキスト(過去の単語)から、この単語を予測すべき」という訓練データの構造を確認
    # for i in range(1, context_size + 1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(context, "--->", desired)

if __name__ == "__main__":
    main()