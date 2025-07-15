# predictor.py

import torch
from transformers import BertTokenizerFast, BertModel
from mBERT_CRF_model import ftmBERTfeatCRFModel
from preprocessing import extract_radicals, extract_pos_tags
from config import (
    MODEL_PATH, BERT_MODEL_PATH, MODEL_STATE_PATH, TOKENIZER_PATH, GEMINI_API_KEY,
    radical_vocab_size, pos_vocab_size, label2id, label_list
)

# 載入 tokenizer 和 BERT model
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)

# 建立模型並載入參數
model = ftmBERTfeatCRFModel(
    model=bert_model,
    num_labels=len(label_list),
    num_radicals=radical_vocab_size,
    num_pos_tags=pos_vocab_size,
    label2id=label2id
)
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=torch.device("cpu")))
model.eval()

def run_prediction(text: str):
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")
    
    tokens = list(text)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    seq_len = encoding.input_ids.shape[1]
    radical_features = extract_radicals(tokens, max_len=seq_len)
    pos_features = extract_pos_tags(text, max_len=seq_len)

    with torch.no_grad():
        predictions = model.decode(
            encoding.input_ids,
            encoding.attention_mask,
            radical_features,
            pos_features
        )
    
    predicted_ids = predictions[0][1:len(tokens)+1]
    final_labels = [label_list[i] for i in predicted_ids]

    return {"tokens": tokens, "labels": final_labels}
