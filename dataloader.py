import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = [] # 入力変数のテンソル
    self.target_ids = [] # 目的変数のテンソル
    token_ids = tokenizer.encode(txt) # テキスト全体をトークン化

    # スライディングウィンドウを使って入力文章を max_length の長さのシーケンスに分割
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  # データセットに含まれている行の総数を返す
  def __len__(self):
    return len(self.input_ids)

  # データセットから1行返す
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
