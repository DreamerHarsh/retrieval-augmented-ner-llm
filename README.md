# Retrieval-Augmented Named Entity Recognition with Large Language Models

A Streamlit app that identifies named entities in text. When run locally, it retrieves similar labeled examples and gives them to a Llama model running through Ollama.

## How it works

1. TF-IDF ranks labeled examples from `data/examples.json` by similarity to the input.
2. The app adds the top examples to a prompt for Llama 3.2 running locally through Ollama.
3. The model returns entities labeled `PERSON`, `ORG`, `LOC`, `DATE`, or `MISC`.
4. The app checks that each returned entity appears in the input text and uses a valid label.

If Ollama is unavailable, the app uses a spaCy NER baseline.

## Run locally

Requirements: Python and [Ollama](https://ollama.com/download).

From the project folder, run these commands in PowerShell:

```powershell
py -m pip install -r requirements.txt
ollama pull llama3.2:3b
py -m streamlit run app.py