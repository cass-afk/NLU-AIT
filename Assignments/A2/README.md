# A3: Make Your Own Machine Translation

This repository contains the implementation of a simple **English to Nepali machine translation web application**, developed as part of **Assignment 3 (A3)**. The application demonstrates how a Transformer-based sequence-to-sequence model with attention can be integrated into an interactive user interface.

---

## Overview

For this assignment, the web interface is built using **Dash**. The **entire UI logic and model integration** are implemented within the `app.py` file. The interface is intentionally kept simple and intuitive, consisting of:

* A text input field for the user query (English sentence)
* A **Translate** button
* Basic input validation
* A section to display the translated Nepali output

A visual demonstration of the interface is provided as a GIF.

---

## Model Integration

The translation model is integrated into the interface through the following steps:

1. **Vocabulary Loading**
   The source and target vocabularies are loaded using `torch.load` from PyTorch.

2. **Model Setup**
   The Transformer model is initialized by passing the required parameters and loading the trained weights from a checkpoint.

3. **Choice of Attention Mechanism**
   Among the three attention variants explored (general, multiplicative, and additive), **additive attention** was selected for deployment. This decision was based on its superior performance in terms of lower validation loss and perplexity during experimentation.

4. **Text Processing Pipeline**
   The input text undergoes the following processing stages:

   * Tokenization
   * Numericalization using the loaded vocabulary
   * Tensor construction with special tokens

5. **Inference and Output Generation**
   During inference, the model predicts the next token iteratively. At each step, the token with the **highest probability** is selected and appended to a list. Once decoding is complete, the tokens are joined to form the final translated sentence, which is then displayed to the user.

---

## User Interaction Flow

The application follows a straightforward interaction flow:

1. The user enters an English sentence into the input field
2. The user clicks the **Translate** button
3. The translated Nepali sentence is displayed on the screen

---

## CPU Deployment Note

The models used in this project were **trained on a GPU**. If the application is run on a **CPU-only environment**, the original GPU-trained checkpoints may cause runtime issues.

To address this, a utility script is provided:

* **`convert_checkpoint.py`**

This script converts GPU-trained checkpoints into **CPU-compatible checkpoints**. It should be executed before running `app.py` when deploying the application on a CPU-only machine.
---
## Demo

The demo video is available here:
![Application Demo](demo-a3.gif)


