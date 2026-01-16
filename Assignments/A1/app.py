from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
import re
import os

# ============================
# Paths
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "model")   # your folder name is "model"
CONTEXTS_PATH = os.path.join(BASE_DIR, "contexts.txt")

# ============================
# Load embedding models
# ============================
embedding_dicts = {}
for model_name in ["glove", "skipgram", "skipgram_negative"]:
    path = os.path.join(MODELS_DIR, f"embed_{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}")
    with open(path, "rb") as f:
        embedding_dicts[model_name] = pickle.load(f)

print("Loaded models:", list(embedding_dicts.keys()))

# ============================
# Load corpus contexts
# ============================
if not os.path.exists(CONTEXTS_PATH):
    raise FileNotFoundError("contexts.txt not found")

with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
    contexts = [line.strip() for line in f if line.strip()]

print("Loaded contexts:", len(contexts))

# ============================
# Tokenization
# ============================
token_re = re.compile(r"[a-z]+")

def tokenize(text: str):
    return token_re.findall(text.lower())

contexts_tokens = [tokenize(c) for c in contexts]

# ============================
# Vector utilities
# ============================
def get_dim(embeddings):
    for v in embeddings.values():
        return int(len(v))
    raise ValueError("Empty embeddings")

def average_vector(tokens, embeddings, dim):
    vecs = []
    for w in tokens:
        if w in embeddings:
            vecs.append(np.asarray(embeddings[w], dtype=np.float32))
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1e9
    return float(np.dot(a, b) / (na * nb))

# Cache context matrices per model
context_cache = {}
def build_context_matrix(model_key):
    if model_key in context_cache:
        return context_cache[model_key]

    embeddings = embedding_dicts[model_key]
    dim = get_dim(embeddings)

    C = np.zeros((len(contexts_tokens), dim), dtype=np.float32)
    valid = np.zeros(len(contexts_tokens), dtype=bool)

    for i, toks in enumerate(contexts_tokens):
        v = average_vector(toks, embeddings, dim)
        if v is not None:
            C[i] = v
            valid[i] = True

    context_cache[model_key] = (C, valid)
    return C, valid

def top_k_contexts_dot(query_vec, C, valid, k=10):
    scores = C @ query_vec
    scores = np.where(valid, scores, -1e9)

    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]

def top_k_words_cosine(query_word, embeddings, k=10):
    if query_word not in embeddings:
        return None  # caller shows message

    qv = embeddings[query_word]
    scored = []
    # This is O(V). OK for assignment size; if your vocab is huge, we can optimize later.
    for w, v in embeddings.items():
        if w == query_word:
            continue
        scored.append((w, cosine_similarity(qv, v)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

# ============================
# Dash App
# ============================
app = Dash(__name__)
app.title = "A1 Search Engine"

model_names = {
    "glove": "GloVe",
    "skipgram": "Skip-gram",
    "skipgram_negative": "Skip-gram (Negative Sampling)"
}

# Simple CSS styling inline
PAGE_STYLE = {
    "minHeight": "100vh",
    "background": "linear-gradient(180deg, #f6f8ff 0%, #ffffff 70%)",
    "padding": "28px 14px",
    "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    "color": "#111827",
}

CARD_STYLE = {
    "maxWidth": "900px",
    "margin": "0 auto",
    "background": "white",
    "borderRadius": "16px",
    "boxShadow": "0 10px 30px rgba(17, 24, 39, 0.08)",
    "border": "1px solid rgba(17, 24, 39, 0.08)",
    "padding": "20px",
}

TITLE_STYLE = {
    "textAlign": "center",
    "margin": "0",
    "fontSize": "28px",
    "fontWeight": "800",
    "letterSpacing": "-0.02em",
}

SUBTITLE_STYLE = {
    "textAlign": "center",
    "marginTop": "8px",
    "marginBottom": "18px",
    "color": "#4b5563",
    "fontSize": "14px",
}

ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "12px",
    "marginTop": "12px",
}

CONTROL_LABEL = {
    "fontSize": "12px",
    "fontWeight": "700",
    "color": "#374151",
    "marginBottom": "6px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "10px 12px",
    "borderRadius": "10px",
    "border": "1px solid rgba(17, 24, 39, 0.18)",
    "outline": "none",
}

BUTTON_STYLE = {
    "padding": "10px 14px",
    "borderRadius": "12px",
    "border": "none",
    "backgroundColor": "#2563eb",
    "color": "white",
    "fontWeight": "800",
    "cursor": "pointer",
    "width": "100%",
}

RESULT_CARD = {
    "padding": "12px 14px",
    "borderRadius": "14px",
    "border": "1px solid rgba(17, 24, 39, 0.08)",
    "backgroundColor": "#ffffff",
    "boxShadow": "0 6px 18px rgba(17, 24, 39, 0.06)",
    "marginBottom": "10px",
}

PILL_TOGGLE_STYLE = {
    "display": "flex",
    "justifyContent": "center",
    "marginTop": "12px",
}

app.layout = html.Div(style=PAGE_STYLE, children=[
    html.Div(style=CARD_STYLE, children=[
        html.H1("A1 Search Engine", style=TITLE_STYLE),
        html.Div("Choose a model and a mode, then search.", style=SUBTITLE_STYLE),

        html.Div(style=ROW_STYLE, children=[
            html.Div(children=[
                html.Div("Model (required)", style=CONTROL_LABEL),
                dcc.Dropdown(
                    id="model-selector",
                    options=[{"label": v, "value": k} for k, v in model_names.items()],
                    placeholder="Select a model...",
                    clearable=True
                ),
            ]),
            html.Div(children=[
                html.Div("Mode (required)", style=CONTROL_LABEL),
                dcc.RadioItems(
                    id="search-mode",
                    options=[
                        {"label": "Top 10 Contexts", "value": "contexts"},
                        {"label": "Top 10 Similar Words", "value": "words"},
                    ],
                    value="contexts",
                    labelStyle={"display": "inline-block", "marginRight": "14px", "fontWeight": "700"},
                    style={"padding": "10px 6px"},
                ),
            ]),
        ]),

        html.Div(style={"marginTop": "12px"}, children=[
            html.Div("Query (required)", style=CONTROL_LABEL),
            dcc.Input(
                id="search-query",
                type="text",
                placeholder="Contexts mode: multi-word query (e.g., oil prices). Words mode: single word (e.g., king).",
                style=INPUT_STYLE
            ),
        ]),

        html.Div(style={"marginTop": "12px"}, children=[
            html.Button("Search", id="search-button", n_clicks=0, style=BUTTON_STYLE)
        ]),

        html.Div(id="search-results", style={"marginTop": "18px"})
    ])
])

# ============================
# Callback
# ============================
@app.callback(
    Output("search-results", "children"),
    Input("search-button", "n_clicks"),
    State("search-query", "value"),
    State("model-selector", "value"),
    State("search-mode", "value"),
)
def search(n_clicks, query, model, mode):
    if n_clicks <= 0:
        return html.Div("Enter a query, select a model, choose a mode, then press Search.",
                        style={"color": "#6b7280", "textAlign": "center"})

    # Required checks
    if not model:
        return html.Div("❌ Please select a model.", style={"color": "#b91c1c", "fontWeight": "800"})
    if not mode:
        return html.Div("❌ Please choose a mode (contexts or words).", style={"color": "#b91c1c", "fontWeight": "800"})
    if not query or not str(query).strip():
        return html.Div("❌ Please enter a query.", style={"color": "#b91c1c", "fontWeight": "800"})

    query = str(query).strip()
    embeddings = embedding_dicts[model]

    # -------- Mode: Similar Words --------
    if mode == "words":
        qword = query.lower().strip()
        # enforce single-word input (optional but helps UX)
        q_tokens = tokenize(qword)
        if len(q_tokens) != 1:
            return html.Div("❌ Words mode expects a single word (e.g., 'king').",
                            style={"color": "#b91c1c", "fontWeight": "800"})

        qword = q_tokens[0]
        top = top_k_words_cosine(qword, embeddings, k=10)
        if top is None:
            return html.Div("❌ Word not in this model vocabulary.", style={"color": "#b91c1c", "fontWeight": "800"})

        return html.Div([
            html.Div([
                html.Div(f"Top 10 words similar to '{qword}'",
                         style={"fontWeight": "900", "fontSize": "16px", "marginBottom": "10px"}),
                html.Div(f"Model: {model_names[model]}",
                         style={"color": "#6b7280", "fontSize": "12px", "marginBottom": "12px"}),
                html.Div([
                    html.Div(style=RESULT_CARD, children=[
                        html.Div(f"#{i+1}  {w}", style={"fontWeight": "800", "fontSize": "14px"}),
                        html.Div(f"Cosine: {score:.6f}", style={"color": "#6b7280", "fontSize": "12px"})
                    ])
                    for i, (w, score) in enumerate(top)
                ])
            ])
        ])

    # -------- Mode: Context Search (Task 3) --------
    # Dot product between query vector and all context vectors
    tokens = tokenize(query)
    if not tokens:
        return html.Div("❌ Query must contain alphabetic characters.", style={"color": "#b91c1c", "fontWeight": "800"})

    dim = get_dim(embeddings)
    query_vec = average_vector(tokens, embeddings, dim)
    if query_vec is None:
        return html.Div("❌ None of the query words exist in this model vocabulary.",
                        style={"color": "#b91c1c", "fontWeight": "800"})

    C, valid = build_context_matrix(model)
    top = top_k_contexts_dot(query_vec, C, valid, k=10)

    return html.Div([
        html.Div(f"Top 10 contexts for '{query}'",
                 style={"fontWeight": "900", "fontSize": "16px", "marginBottom": "10px"}),
        html.Div(f"Model: {model_names[model]} | Similarity: dot product",
                 style={"color": "#6b7280", "fontSize": "12px", "marginBottom": "12px"}),

        html.Div([
            html.Div(style=RESULT_CARD, children=[
                html.Div(f"#{rank}  Score: {score:.6f}",
                         style={"fontWeight": "800", "fontSize": "13px", "marginBottom": "6px"}),
                html.Div(contexts[i], style={"fontSize": "14px", "lineHeight": "1.35"})
            ])
            for rank, (i, score) in enumerate(top, start=1)
        ])
    ])

# ============================
# Run
# ============================
if __name__ == "__main__":
    app.run(debug=True)
