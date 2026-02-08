import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import torch
from torchtext.data.utils import get_tokenizer

from nepalitokenizers import WordPiece
from S2S import Seq2SeqTransformer

# ----------------------------
# Device (CPU/GPU safe)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARG_LANG = "ne"
SRC_LANG  = "en"

MAX_DECODE_LEN = 60  # stop after this many tokens (prevents infinite loops)

# ----------------------------
# Paths (robust)
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.normpath(os.path.join(current_dir, "..", "model", "vocab"))

# CLEAN checkpoint (dict with {"params":..., "state_dict":...})
model_path = os.path.normpath(
    os.path.join(current_dir, "..", "model", "multiplicative_Seq2SeqTransformer.clean.pt")
)

# ----------------------------
# Load vocab (CPU safe)
# ----------------------------
vocab_transform = torch.load(vocab_path, map_location="cpu")

# ----------------------------
# Tokenizers
# ----------------------------
token_transform = {}

try:
    token_transform["en"] = get_tokenizer("spacy", language="en_core_web_sm")
except Exception:
    token_transform["en"] = get_tokenizer("basic_english")

token_transform["ne"] = WordPiece()

# ----------------------------
# Special token ids
# ----------------------------
def get_vocab_index(vocab, token, fallback=None):
    try:
        return vocab[token]
    except Exception:
        if fallback is None:
            raise
        return fallback

SRC_VOCAB = vocab_transform[SRC_LANG]
TRG_VOCAB = vocab_transform[TARG_LANG]

SRC_SOS_IDX = get_vocab_index(SRC_VOCAB, "<sos>", fallback=2)
SRC_EOS_IDX = get_vocab_index(SRC_VOCAB, "<eos>", fallback=3)

TRG_SOS_IDX = get_vocab_index(TRG_VOCAB, "<sos>", fallback=2)
TRG_EOS_IDX = get_vocab_index(TRG_VOCAB, "<eos>", fallback=3)
TRG_PAD_IDX = get_vocab_index(TRG_VOCAB, "<pad>", fallback=1)

# ----------------------------
# Transform pipeline
# ----------------------------
def sequential_operation(*transforms):
    """
    Applies transforms in order.
    WordPiece tokenizer case: object has .encode().tokens
    """
    def text_operation(input_text):
        x = input_text
        for transform in transforms:
            if hasattr(transform, "encode") and callable(getattr(transform, "encode")):
                x = transform.encode(x).tokens
            else:
                x = transform(x)
        return x
    return text_operation

def tensor_transform(token_ids, sos_idx, eos_idx):
    return torch.cat((
        torch.tensor([sos_idx], dtype=torch.long),
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor([eos_idx], dtype=torch.long),
    ))

text_transform = {}
text_transform[SRC_LANG] = sequential_operation(
    token_transform[SRC_LANG],
    vocab_transform[SRC_LANG],
    lambda ids: tensor_transform(ids, SRC_SOS_IDX, SRC_EOS_IDX)
)
text_transform[TARG_LANG] = sequential_operation(
    token_transform[TARG_LANG],
    vocab_transform[TARG_LANG],
    lambda ids: tensor_transform(ids, TRG_SOS_IDX, TRG_EOS_IDX)
)

# ----------------------------
# Load model (clean checkpoint)
# ----------------------------
ckpt = torch.load(model_path, map_location=device)
params = ckpt["params"]
state  = ckpt["state_dict"]

# Force device consistency in params (important if saved from GPU)
if isinstance(params, dict):
    params["device"] = device

model = Seq2SeqTransformer(**params).to(device)
model.load_state_dict(state)
model.eval()

# ----------------------------
# Greedy decoding (proper translation)
# ----------------------------
def greedy_translate(src_text: str) -> str:
    src_tensor = text_transform[SRC_LANG](src_text.strip()).to(device).unsqueeze(0)

    trg_ids = [TRG_SOS_IDX]

    with torch.no_grad():
        for _ in range(MAX_DECODE_LEN):
            trg_tensor = torch.tensor([trg_ids], dtype=torch.long, device=device)
            output, _ = model(src_tensor, trg_tensor)

            next_id = int(output[0, -1].argmax(-1).item())
            trg_ids.append(next_id)

            if next_id == TRG_EOS_IDX:
                break

    itos = TRG_VOCAB.get_itos()
    toks = []
    for idx in trg_ids:
        tok = itos[idx] if idx < len(itos) else "<unk>"
        if tok in ["<sos>", "<eos>", "<pad>", "[CLS]", "[SEP]", "[EOS]"]:
            continue
        toks.append(tok)

    return " ".join(toks) if toks else "(no output)"

# ----------------------------
# Dash app (prettier UI, no assets)
# ----------------------------
# ----------------------------
# Dash app (enhanced colorful UI)
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        # ===== Header =====
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H1("🌐 English → Nepali Translator", className="fw-bold mb-1"),
                            html.P(
                                "Transformer with Multiplicative Attention",
                                className="mb-0",
                                style={"opacity": "0.9"},
                            ),
                        ],
                        className="text-white text-center",
                    ),
                    style={
                        "background": "linear-gradient(135deg, #4f46e5, #9333ea)",
                        "border": "none",
                    },
                    className="shadow-lg",
                ),
                width=10,
            ),
            justify="center",
            className="mt-4",
        ),

        # ===== Main Card =====
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    # ----- INPUT -----
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H5("✍️ Input (English)", className="fw-semibold"),
                                                    dbc.Textarea(
                                                        id="user-input",
                                                        placeholder="Type an English sentence here…",
                                                        style={
                                                            "height": "140px",
                                                            "resize": "none",
                                                        },
                                                    ),
                                                    dbc.Button(
                                                        "🚀 Translate",
                                                        id="translate-btn",
                                                        color="primary",
                                                        className="w-100 mt-3",
                                                    ),
                                                    dbc.Button(
                                                        "🧹 Clear",
                                                        id="clear-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="w-100 mt-2",
                                                    ),
                                                ]
                                            ),
                                            className="border-0 shadow-sm",
                                            style={"backgroundColor": "#eef2ff"},
                                        ),
                                        md=6,
                                    ),

                                    # ----- OUTPUT -----
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H5("📜 Output (Nepali)", className="fw-semibold"),
                                                    dbc.Spinner(
                                                        html.Div(
                                                            id="translation-output",
                                                            className="fs-5",
                                                            style={
                                                                "whiteSpace": "pre-wrap",
                                                                "minHeight": "100px",
                                                            },
                                                        ),
                                                        color="primary",
                                                    ),
                                                    dbc.Badge(
                                                        "Greedy Decoding",
                                                        color="info",
                                                        className="mt-3 me-2",
                                                    ),
                                                    dbc.Badge(
                                                        f"Max Len = {MAX_DECODE_LEN}",
                                                        color="warning",
                                                        className="mt-3",
                                                    ),
                                                ]
                                            ),
                                            className="border-0 shadow-sm",
                                            style={"backgroundColor": "#f0fdf4"},
                                        ),
                                        md=6,
                                    ),
                                ],
                                className="g-4",
                            ),
                        ]
                    ),
                    className="border-0 shadow-lg mt-4",
                ),
                width=10,
            ),
            justify="center",
        ),

        # ===== Footer =====
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.Hr(),
                        html.Small(
                            "💡 Tip: Short, simple sentences usually translate better with greedy decoding.",
                            className="text-muted",
                        ),
                        html.Br(),
                        html.Small(
                            f"Running on device: {device}",
                            className="text-muted",
                        ),
                    ],
                    className="text-center",
                ),
                width=10,
            ),
            justify="center",
            className="mb-4",
        ),
    ],
    fluid=True,
)

# ===== Callbacks =====
@app.callback(
    Output("user-input", "value"),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_input(_):
    return ""

@app.callback(
    Output("translation-output", "children"),
    Input("translate-btn", "n_clicks"),
    State("user-input", "value"),
)
def translate_text(n_clicks, text):
    if not n_clicks:
        return ""

    if not text or not text.strip():
        return "⚠️ Please enter some text."

    return greedy_translate(text)

if __name__ == "__main__":
    app.run(debug=True)
