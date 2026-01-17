# A1 – Word Embedding Search Engine

This project implements multiple word embedding models (Skip-gram, Skip-gram with Negative Sampling, and GloVe) and a simple web-based search engine built with Dash.  
The web application allows users to search for **similar contexts** or **similar words** using the trained embeddings.


---

## 1. Train and Generate the Models (Required)

Before running the web application, you must first execute the provided Jupyter Notebook (`.ipynb`) files.

These notebooks:
- train the word embedding models on the **NLTK Brown news corpus**
- save the trained embeddings as `.pkl` files inside the `model/` directory


---

## 2. Install Required Libraries

Install all required Python dependencies using:

pip install -r requirements.txt

## 3. Running the app

From the project root directory, start the Dash application using:

1. python app.py
2. Open the Web Interface on http://127.0.0.1:8050 
3. Select the model
4. Enter the Word
5. Select if you want to generate the first 10 contexts or 10 words

## Demo

![Application Demo](demo-A1.gif)
