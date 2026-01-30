from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
import torch, torchtext
from torchtext.data.utils import get_tokenizer
from lstm import LSTMLanguageModel
import os

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a basic English tokenizer from torchtext
tokenizer = get_tokenizer("basic_english")

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the vocabulary
model_path = os.path.join(current_dir, "model\\vocab_lm.pkl")
with open(model_path, "rb") as f:
    loaded_vocab = pickle.load(f)

# Path to the trained model
model_path_2 = os.path.join(current_dir, "model\\best-val-lstm_lm.pt")

# Same hyperparameters as used during training
vocab_size = len(loaded_vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65

lstm_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
lstm_model.load_state_dict(torch.load(model_path_2, map_location=device))


def generate_text(prompt, max_seq, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for _ in range(max_seq):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)

            probability = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probability, num_samples=1).item()

            while prediction == vocab["<unk>"]:  # sample again if <unk>
                prediction = torch.multinomial(probability, num_samples=1).item()

            if prediction == vocab["<eos>"]:  # stop if <eos>
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


app = Dash(__name__)
app.title = "A2 Language Model"

THEME = {
    "bg": "#0b1020",
    "panel": "#111a33",
    "card": "#0f1730",
    "border": "rgba(255,255,255,0.10)",
    "text": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.65)",
    "accent": "#6ea8fe",
    "accent2": "#9b7bff",
    "danger": "#ff6b6b",
}


def pill(text):
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "6px 10px",
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "999px",
            "fontSize": "12px",
            "color": THEME["muted"],
            "marginRight": "8px",
            "marginBottom": "8px",
            "background": "rgba(255,255,255,0.03)",
        },
    )


app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": (
            "radial-gradient(1200px 600px at 20% 0%, rgba(110,168,254,0.20), transparent),"
            "radial-gradient(1200px 600px at 80% 10%, rgba(155,123,255,0.18), transparent),"
            f"{THEME['bg']}"
        ),
        "padding": "36px 16px",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
        "color": THEME["text"],
    },
    children=[
        html.Div(
            style={"maxWidth": "980px", "margin": "0 auto"},
            children=[
                # Header
                html.Div(
                    style={"marginBottom": "18px"},
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "flex-end",
                                "gap": "12px",
                                "flexWrap": "wrap",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.H1(
                                            "A2: Language Model",
                                            style={"margin": "0", "fontSize": "34px", "letterSpacing": "-0.5px"},
                                        ),
                                        html.P(
                                            "Generate continuations with different sampling temperatures.",
                                            style={"margin": "6px 0 0", "color": THEME["muted"]},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "8px",
                                        "flexWrap": "wrap",
                                        "justifyContent": "flex-end",
                                    },
                                    children=[
                                        pill("LSTM LM"),
                                        pill("Torch + TorchText"),
                                        pill("GPU" if torch.cuda.is_available() else "CPU"),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                # Controls panel
                html.Div(
                    style={
                        "background": THEME["panel"],
                        "border": f"1px solid {THEME['border']}",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.35)",
                    },
                    children=[
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1.6fr 1fr", "gap": "12px"},
                            children=[
                                # Prompt input
                                html.Div(
                                    children=[
                                        html.Label("Prompt", style={"fontSize": "13px", "color": THEME["muted"]}),
                                        dcc.Input(
                                            id="search-query",
                                            type="text",
                                            placeholder="Type a prompt… (press Enter to generate)",
                                            debounce=True,
                                            style={
                                                "width": "100%",
                                                "padding": "12px 12px",
                                                "marginTop": "8px",
                                                "borderRadius": "12px",
                                                "border": f"1px solid {THEME['border']}",
                                                "background": "rgba(255,255,255,0.03)",
                                                "color": THEME["text"],
                                                "outline": "none",
                                            },
                                        ),
                                    ]
                                ),
                                # Right side control: Max tokens only (seed removed)
                                html.Div(
                                    children=[
                                        html.Label("Max tokens", style={"fontSize": "13px", "color": THEME["muted"]}),
                                        dcc.Slider(
                                            id="max-seq",
                                            min=5,
                                            max=80,
                                            step=1,
                                            value=30,
                                            marks={5: "5", 30: "30", 80: "80"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(style={"height": "10px"}),
                        # Temps picker + button row
                        html.Div(
                            style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        html.Label("Temperatures", style={"fontSize": "13px", "color": THEME["muted"]}),
                                        dcc.Checklist(
                                            id="temps",
                                            options=[
                                                {"label": "0.5", "value": 0.5},
                                                {"label": "0.7", "value": 0.7},
                                                {"label": "0.75", "value": 0.75},
                                                {"label": "0.8", "value": 0.8},
                                                {"label": "1.0", "value": 1.0},
                                                {"label": "1.2", "value": 1.2},
                                            ],
                                            value=[0.5, 0.7, 0.75, 0.8, 1.0],
                                            inline=True,
                                            style={"marginTop": "8px"},
                                            labelStyle={
                                                "marginRight": "12px",
                                                "padding": "6px 10px",
                                                "borderRadius": "999px",
                                                "border": f"1px solid {THEME['border']}",
                                                "background": "rgba(255,255,255,0.03)",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Generate",
                                    id="search-button",
                                    n_clicks=0,
                                    style={
                                        "padding": "12px 16px",
                                        "borderRadius": "12px",
                                        "border": "none",
                                        "cursor": "pointer",
                                        "color": "#0b1020",
                                        "fontWeight": 700,
                                        "background": f"linear-gradient(135deg, {THEME['accent']}, {THEME['accent2']})",
                                        "minWidth": "140px",
                                    },
                                ),
                                dcc.Loading(
                                    type="dot",
                                    children=html.Div(id="status", style={"color": THEME["muted"], "fontSize": "13px"}),
                                ),
                            ],
                        ),
                    ],
                ),
                # Results
                html.Div(style={"height": "18px"}),
                dcc.Loading(type="default", children=html.Div(id="search-results")),
                html.Div(
                    style={"marginTop": "18px", "color": THEME["muted"], "fontSize": "12px"},
                    children="Tip: Lower temperature = safer/more repetitive. Higher temperature = more diverse/creative.",
                ),
            ],
        )
    ],
)


@app.callback(
    Output("search-results", "children"),
    Output("status", "children"),
    Input("search-button", "n_clicks"),
    Input("search-query", "value"),  # Enter-to-generate (debounce=True)
    State("max-seq", "value"),
    State("temps", "value"),
    prevent_initial_call=True,
)
def search(n_clicks, query, max_seq_len, temps):
    if not query or not query.strip():
        return (
            html.Div("Please enter a prompt to generate text.", style={"color": THEME["danger"], "padding": "12px"}),
            "Waiting for input…",
        )

    if not temps:
        temps = [1.0]

    # Choose seeding behavior:
    # seed_val = 122   # fixed (reproducible)
    seed_val = None    # random (different each run)

    results = []
    for temperature in temps:
        tokens = generate_text(
            prompt=query.strip(),
            max_seq=max_seq_len,
            temperature=float(temperature),
            model=lstm_model,
            tokenizer=tokenizer,
            vocab=loaded_vocab,
            device=device,
            seed=seed_val,
        )

        results.append(
            html.Div(
                style={
                    "background": THEME["card"],
                    "border": f"1px solid {THEME['border']}",
                    "borderRadius": "16px",
                    "padding": "14px 14px",
                    "boxShadow": "0 10px 26px rgba(0,0,0,0.25)",
                },
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "gap": "10px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Div(
                                        f"Temperature {temperature}",
                                        style={"fontWeight": 800, "fontSize": "14px"},
                                    ),
                                    html.Div(
                                        f"max={max_seq_len}",
                                        style={"color": THEME["muted"], "fontSize": "12px", "marginTop": "2px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(style={"height": "10px"}),
                    html.Pre(
                        " ".join(tokens),
                        style={
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-word",
                            "margin": 0,
                            "fontSize": "14px",
                            "lineHeight": "1.5",
                            "color": THEME["text"],
                            "background": "rgba(255,255,255,0.02)",
                            "border": f"1px solid {THEME['border']}",
                            "borderRadius": "12px",
                            "padding": "12px",
                        },
                    ),
                ],
            )
        )

    grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
            "gap": "12px",
        },
        children=results,
    )

    return grid, f"Generated {len(results)} sample(s)."


if __name__ == "__main__":
    app.run(debug=True)
