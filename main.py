import urllib.request
from simple_tokenizer import SimpleTokenizerV1          

def retrieve_raw_text():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt"
           )
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def main():
    # retrieve_raw_text()
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        tokenizer = SimpleTokenizerV1(raw_text)
    

if __name__ == "__main__":
    main()