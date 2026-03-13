import urllib.request
import re
from simple_tokenizer import SimpleTokenizerV1          

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

def main():
    # retrieve_raw_text()
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        vocab = parse_token(raw_text)
        tokenizer = SimpleTokenizerV1(vocab)
        print(tokenizer.encode('Hello, do you wanna eat lunch'))

if __name__ == "__main__":
    main()