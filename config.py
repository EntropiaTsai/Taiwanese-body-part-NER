import os
import json

# ✅ 開發階段讀入 .env
if os.getenv("ENV", "local") == "local":
    from dotenv import load_dotenv
    load_dotenv()

# ========= 模型與詞表設定 =========
MODEL_PATH = "./model_state"
BERT_MODEL_PATH = "./mBERT_model"
MODEL_STATE_PATH = f"{MODEL_PATH}/model_state_dict.pt"
TOKENIZER_PATH = MODEL_PATH

label_list = ["B-BODY", "I-BODY", "O"]
label2id = {label: idx for idx, label in enumerate(label_list)}

with open("./feature_resources/radical2id.json", "r") as f:
    radical2id = json.load(f)
with open("./feature_resources/pos2id.json", "r") as f:
    pos2id = json.load(f)

radical_vocab_size = len(radical2id)
pos_vocab_size = len(pos2id)

# ========= API 金鑰設定 =========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("❌ 請設定 GEMINI_API_KEY，建議使用 .env 或環境變數")
