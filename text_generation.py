from config import GEMINI_API_KEY
import requests
import re
import json

# Gemini API 設定
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print("❌ Gemini 錯誤：", e)
        return "抱歉，我暫時無法回應。"

one_span={"results": [ {"tokens": [ "伊", "的", "面", "皮", "足", "白", "。"], "labels": ["O", "O", "B-BODY", "I-BODY", "O", "O", "O"]}]}
multiple_span={"results": [ {"tokens": [ "伊", "的", "面", "皮", "足", "白", "。"], "labels": ["O", "O", "B-BODY", "B-BODY", "O", "O", "O"]}]}


metaphorical={"results": [ {"tokens": [ "伊", "的", "面", "皮", "足", "薄", "。"], "labels": ["O", "O", "O", "O", "O", "O", "O"]}]}
literal={"results": [ {"tokens": [ "伊", "的", "面", "皮", "足", "白", "。"], "labels": ["O", "O", "B-BODY", "I-BODY", "O", "O", "O"]}]}

def gereration(predicted_labels):
     # Prompt
    prompt = f"""
    你是一位專業語言學家，請根據以下標記結果進行校正。

    以下是模型自動標記的結果：
    {predicted_labels}

    ---

    🔍 校正目標：

    1️⃣ **修正標記範圍（Span）**

    請檢查是否有**語意上屬於同一身體部位實體**，但被標記為連續多個 `B-BODY`。

    - ✅ 正確範圍（應該合併成一個實體）：
    *面皮* 是一個連續詞組，代表同一個實體，應標記為：
    `["B-BODY", "I-BODY"]`

    - ❌ 錯誤範圍（被錯誤切開）：
    `["B-BODY", "B-BODY"]` → 表示模型誤把一個實體拆開

    但也有例外：

    - ✅ 正確分開範圍：
    「手腳並用」中的 *手* 和 *腳* 是兩個獨立實體，即使相連，也應分別標記為：
    `["B-BODY", "B-BODY"]`

    👉 所以是否合併，取決於它們**是否共同構成一個語意單位的身體部位名稱**。

    2️⃣ **判斷語意用法（Literal vs Metaphorical）**

    預測中可能包含**實指**身體部位或**比喻用法**。

    - ✅ **實指身體部位**：
    - *面皮*，表示「臉部的皮膚」
    - 標記為：`["B-BODY", "I-BODY"]`
    - 範例：{literal}

    - ❌ **比喻用法**：
    - *面皮*，用來描述「性格」
    - 應標記為：`["O", "O"]`
    - 範例：{metaphorical}

    ---
    📌 請你嚴格遵循以下輸出格式（即使沒有修改也請完整填寫）：

    **修正前的標記結果：**
    ```json
    {json.dumps(predicted_labels, ensure_ascii=False, indent=2)}
    ```

    **修正後的標記結果：**
    ```json
    [
      {{
        "tokens": [...],
        "labels": [...]
      }},
      ...
    ]
    ```

    **修正說明：**
    - 若有修正，請說明調整哪些詞為實指或比喻
    - 若無修正，請說明「標記已正確，無需更動」

    📌 **現在請你依據這兩個任務，修正上方的 predicted_labels。**

    - 合併錯誤切割的 B-BODY span
    - 將比喻用法的 B/I-BODY 標記改為 O

    
    
    """

    answer = call_gemini(prompt)
    return answer


def extract_json_block(gemini_text: str):
    try:
        # 嘗試找出 ```json ``` 區塊
        match = re.search(
            r"\*\*修正後的標記結果：\*\*\s*```json\s*(\{.*?\}|\[.*?\])\s*```",
            gemini_text,
            re.DOTALL
        )        
        if match:
            json_str = match.group(1)
        else:
            # 如果沒有 ```json``` 包起來的內容，試著抓 array 或 dict
            match = re.search(r"(\{.*?\}|\[.*?\])", gemini_text, re.DOTALL)
            if not match:
                return None
            json_str = match.group(1)

        # 修正常見格式錯誤：單引號換成雙引號
        json_str = json_str.replace("'", '"')

        # 嘗試 parse
        parsed = json.loads(json_str)

        # 包裝成 {"results": [...]} 格式
        if isinstance(parsed, list):  # 單純 list
            return {"results": parsed}
        elif isinstance(parsed, dict) and "results" in parsed:
            return parsed
        else:
            return None
    except Exception as e:
        print("❌ JSON 擷取失敗：", e)
        return None