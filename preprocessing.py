# preprocessing.py

import json
import torch
from typing import List
from ckiptagger import WS, POS

# 載入必要資源
with open("./feature_resources/radical2id.json", "r") as f:
    radical2id = json.load(f)

with open("./feature_resources/pos2id.json", "r") as f:
    pos2id = json.load(f)

with open("./feature_resources/Dictionary.json", "r") as f:
    radical_dict = json.load(f)

# 初始化 CKIPTagger（需放入 ckpt 路徑）
ws = WS("./feature_resources/data")
pos_tagger = POS("./feature_resources/data")

def extract_radicals(chars: List[str], max_len: int) -> torch.Tensor:
    ids = []
    for char in chars:
        radical = radical_dict.get(char, {}).get("radical")
        if radical:
            ids.append(radical2id.get(radical, radical2id["<UNK>"]))
        else:
            ids.append(radical2id["<UNK>"])
    ids = [0] + ids[:max_len - 2] + [0]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0)

def extract_pos_tags(text: str, max_len: int) -> torch.Tensor:
    words = ws([text])[0]
    pos_tags = pos_tagger([words])[0]
    tag_ids = [pos2id.get(tag, pos2id["<UNK>"]) for tag in pos_tags]
    ids = [0] + tag_ids[:max_len - 2] + [0]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0)
