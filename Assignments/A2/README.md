# A2 – Language Model Demo

This repository contains a trained **LSTM-based Language Model** and a **Dash web application** for interactive text generation with different sampling temperatures.

The demo showcases how temperature and sequence length affect generated text.

---

## Prerequisites


- PyTorch & TorchText (**must be compatible versions**)
- Dash

---

## Important: Torch & TorchText Compatibility

Make sure your `torch` and `torchtext` versions are compatible.  
A recommended stable pairing is:

```bash
pip install torch==2.1.0 torchtext==0.16.0
```

If you encounter NumPy or TorchText errors, downgrade NumPy as well:

```bash
pip install "numpy<2"
```

---

## How to Run the Project

### **Step 1: Run the Jupyter Notebook**

First, run the notebook to generate all required artifacts (model weights, vocabulary, etc.):

```bash
main.ipynb
```

This notebook will:
- Train / load the LSTM language model
- Save the model checkpoint (`.pt`)
- Save the vocabulary (`.pkl`)

**Do not skip this step**, or the web app will not run.

---

### **Step 2: Launch the Dash Web App**

After the notebook finishes successfully, run:

```bash
python app.py
```

The application will be deployed locally at:

```
http://127.0.0.1:8050/
```

Open this URL in your browser.

---

## Using the Web Interface

Once the app is running, you can:

1. **Enter a text prompt**
2. **Select one or more temperatures**
   - Lower temperature → safer, more repetitive output  
   - Higher temperature → more diverse, creative output
3. **Select maximum number of tokens**
4. Click **Generate**

The app will display generated text for each selected temperature in separate output cards.

---

## 🎥 Demo

The demo video is available here:
![Application Demo](demo-A2.gif)


