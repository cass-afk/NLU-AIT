# A4 – Do you AGREE?

This project implements a **custom BERT model from scratch**, evaluates it on Natural Language Inference (NLI) tasks, and compares it against a **pre-trained sentence embedding model**.

The system is deployed via a **Dash web interface** where users can input two sentences and receive a similarity-based logical classification (Entailment / Neutral / Contradiction).


---

# How to Run the Project

## Run Training & Evaluation Notebooks (Required First)

Before launching the web application, execute the following notebooks:

### Step 1 – Pretrain BERT
Run:

```
task1.ipynb
```

This notebook:
- Implements BERT from scratch
- Trains using Masked Language Modeling (MLM)
- Uses filtered Wikipedia dataset (10k–100k samples depending on memory)

---

### Step 2 – Fine-tuning & Evaluation
Run:

```
t2_t3.ipynb
```

This notebook:
- Fine-tunes the custom BERT on SNLI/MNLI
- Evaluates performance
- Generates classification report
- Compares with a pre-trained model

---

## Launch the Web Application

After running the notebooks:

```bash
python app.py
```

The app will launch at:

```
http://127.0.0.1:8050/
```

---

# How to Use the Web Interface

1. Enter **Sentence 1**
2. Enter **Sentence 2**
3. Click **Analyze**
4. The system will display:
   - Cosine similarity score
   - Predicted relationship:
     - ✅ Entailment
     - ⚪ Neutral
     - ❌ Contradiction

The classification is determined by similarity thresholds:
- ≥ 0.75 → Entailment
- 0.40 – 0.75 → Neutral
- < 0.40 → Contradiction

---

# Model Evaluation

## Evaluation of custom model


| Model Type  | Training Loss with SNLI and MNLI | Cosine Similarity(SNLI and MNLI) | Cosine Similarity (Similar Sentences) | Cosine Similarity (Dissimilar Sentences) |
|------------|------------------------------|----------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Custom model   | 1.89                        | 0.9983                                         | 0.9989                                             | 0.9989                                             |



## Classification Report (Custom Model)

| Class           | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Entailment     | 0.34      | 0.99   | 0.51     | 338     |
| Neutral        | 0.43      | 0.02   | 0.04     | 328     |
| Contradiction  | 0.00      | 0.00   | 0.00     | 334     |
| **Accuracy**   | —         | —      | **0.34** | 1000    |
| **Macro Avg**  | 0.26      | 0.34   | 0.18     | 1000    |
| **Weighted Avg** | 0.26    | 0.34   | 0.18     | 1000    |


---

# Comparison with Pre-trained Model

We compared our custom model with a pre-trained sentence embedding model (`all-mpnet-base-v2`).

## Comparison of custom model with pre-trained model

| Model Type     | Similar Sentences | Dissimilar Sentences |
|---------------|------------------|----------------------|
| **Custom Model** | 0.9992           | 0.9990               |
| **Pre-trained Model** | 0.731            | 0.483                |


---
 
# DEMO

![Application Demo](demo-a4.gif)

