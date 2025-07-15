import gradio as gr
import requests

API_URL = "http://127.0.0.1:8080/predict_batch"

def call_api(texts_input):
    texts = [line.strip() for line in texts_input.strip().split("\n") if line.strip()]
    payload = {"texts": texts}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])

        # 包裝一層 scrollable div
        output_html = "<div style='overflow-x: auto; white-space: nowrap;'>"

        for item in results:
            tokens = item["tokens"]
            labels = item["labels"]

            output_html += "<div style='line-height: 1.6em; font-family: monospace;'>"
            # 一行 token
            for tok in tokens:
                output_html += f"<span style='display:inline-block; width:6em; text-align:center;'>{tok}</span>"
            output_html += "<br>"

            # 一行標籤
            for lab in labels:
                if lab == "B-BODY":
                    color = "red"
                elif lab == "I-BODY":
                    color = "green"
                else:
                    color = "gray"
                output_html += f"<span style='display:inline-block; width:6em; text-align:center; color:{color};'>{lab}</span>"

            output_html += "</div><br>"

        output_html += "</div>"  # 關閉 scrollable div

        return output_html

    except Exception as e:
        return f"<p style='color:red;'>❌ 發生錯誤：{str(e)}</p>"

iface = gr.Interface(
    fn=call_api,
    inputs=gr.Textbox(lines=5, placeholder="請輸入台語句子，每行一句...", label="輸入台語文本"),
    outputs=gr.HTML(label="標註結果"),
    title="台語身體部位自動標註",
    description="輸入一句或多句台語句子，每行一句，系統會進行標註其中的身體部位"
)

if __name__ == "__main__":
    iface.launch()
