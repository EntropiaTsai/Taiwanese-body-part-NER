## 📘 專案說明（繁體中文）

本專案為一套針對 **台語文本中身體部位詞** 的命名實體辨識（Named Entity Recognition, NER）系統範本，並透過 Gradio 介面呈現。

- Embeddings: 
(由於容量限制，本專案為上傳提取 embeddings 之相關模型及詞表)
    - POS tagging (CKIP) 
    - 部首
    - mBERT embeddings （使用部分資料微調） 

- Model:
    - CRF 進行初步標記
    - Gemini 排除隱喻性用法
        - 字面用法：`伊的面皮足薄。`
        - 隱喻用法：`伊的面皮足白。`

- Demo:
    - `app.py` 透過 FastAPI 產生 API 後，導入 `gradio_ui.py` 產生使用者介面。

## 📁 使用方法
- 透過 uvicorn 啟動 `app.py` 產生 API:
`uvicorn app:app --reload --port 8000`

- 運行 `gradio_ui.py` 啟動使用者介面。

`python gradio_ui.py`# Taiwanese-body-part-NER
