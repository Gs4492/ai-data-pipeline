from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    df = pd.read_csv(file)

    summary = df.describe().to_string()

    prompt = f"""
Here is dataset statistics:

{summary}

Generate business insights and trends.
"""

    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.3",
        messages=[{"role":"user","content":prompt}],
        max_tokens=700
    )

    output = completion.choices[0].message.content
    return jsonify({"insights": output})

if __name__ == "__main__":
    app.run(debug=True)
