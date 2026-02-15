"""Dash UI for: A4: Do you Agree?
"""

from __future__ import annotations

import os
import torch
from dash import Dash, html, dcc, Input, Output, State, no_update

from transformers import BertTokenizer
from Bert import BERT, calculate_similarity


# -----------------------------
# Model / Tokenizer Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model", "sen_bert_cpu.pth")

params, state = torch.load(model_path, map_location=device)
model_bert = BERT(**params, device=device).to(device)
model_bert.load_state_dict(state)
model_bert.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# -----------------------------
# UI Helpers
# -----------------------------
def classify_from_score(score: float) -> tuple[str, str]:
    """
    Returns (label, color).
    """
    if score >= 0.75:
        return "Entailment", "#16a34a"  # green
    if score < 0.40:
        return "Contradiction", "#dc2626"  # red
    return "Neutral", "#f59e0b"  # amber


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def result_card(label: str, color: str, score: float, q1: str, q2: str) -> html.Div:
    score01 = clamp01(score)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Prediction", style={"fontSize": "12px", "opacity": "0.75"}),
                            html.Div(
                                label,
                                style={
                                    "display": "inline-block",
                                    "padding": "6px 10px",
                                    "borderRadius": "999px",
                                    "backgroundColor": f"{color}1A",  # translucent
                                    "border": f"1px solid {color}55",
                                    "color": color,
                                    "fontWeight": "700",
                                    "marginTop": "6px",
                                },
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [
                            html.Div("Cosine similarity", style={"fontSize": "12px", "opacity": "0.75"}),
                            html.Div(
                                f"{score:.4f}",
                                style={"fontSize": "20px", "fontWeight": "800", "marginTop": "6px"},
                            ),
                        ],
                        style={"textAlign": "right"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        style={
                            "height": "10px",
                            "borderRadius": "999px",
                            "backgroundColor": "#e5e7eb",
                            "overflow": "hidden",
                        }
                    ),
                    html.Div(
                        style={
                            "height": "10px",
                            "marginTop": "-10px",
                            "width": f"{score01 * 100:.2f}%",
                            "borderRadius": "999px",
                            "backgroundColor": color,
                            "transition": "width 250ms ease",
                        }
                    ),
                ],
                style={"marginTop": "16px"},
            ),
            html.Details(
                [
                    html.Summary("View input sentences"),
                    html.Div(
                        [
                            html.Div("Sentence 1", style={"fontSize": "12px", "opacity": "0.75", "marginTop": "8px"}),
                            html.Div(q1, style={"padding": "10px", "background": "#f9fafb", "borderRadius": "10px"}),
                            html.Div("Sentence 2", style={"fontSize": "12px", "opacity": "0.75", "marginTop": "12px"}),
                            html.Div(q2, style={"padding": "10px", "background": "#f9fafb", "borderRadius": "10px"}),
                        ],
                        style={"marginTop": "10px"},
                    ),
                ],
                style={"marginTop": "14px"},
            ),
        ],
        style={
            "backgroundColor": "white",
            "border": "1px solid #e5e7eb",
            "borderRadius": "16px",
            "padding": "18px",
            "boxShadow": "0 10px 25px rgba(0,0,0,0.08)",
            "maxWidth": "760px",
            "width": "100%",
        },
    )


# -----------------------------
# App Setup
# -----------------------------
app = Dash(__name__)
app.title = "A4: Do you Agree?"

APP_BG = "#0b1220"   # deep navy
CARD_BG = "#0f172a"  # slate
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
BORDER = "rgba(148,163,184,0.18)"

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "A4: Do you Agree?",
                            style={"fontSize": "34px", "fontWeight": "900", "letterSpacing": "-0.5px"},
                        ),
                        html.Div(
                            "Enter two sentences to estimate similarity and infer a relationship label.",
                            style={"marginTop": "8px", "color": MUTED, "fontSize": "14px"},
                        ),
                        html.Div(
                            f"Running on: {device.type.upper()}",
                            style={"marginTop": "10px", "color": MUTED, "fontSize": "12px"},
                        ),
                    ],
                    style={"maxWidth": "980px", "margin": "0 auto", "padding": "32px 18px 16px 18px"},
                )
            ],
            style={"borderBottom": f"1px solid {BORDER}"},
        ),

        # Main
        html.Div(
            [
                html.Div(
                    [
                        # Input Card
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Compare two sentences",
                                            style={"fontSize": "16px", "fontWeight": "800", "color": TEXT},
                                        ),
                                        html.Div(
                                            "Tip: press Enter in either box to run.",
                                            style={"fontSize": "12px", "color": MUTED, "marginTop": "4px"},
                                        ),
                                    ]
                                ),

                                html.Div(style={"height": "14px"}),

                                html.Label("Sentence 1", style={"color": MUTED, "fontSize": "12px"}),
                                dcc.Textarea(
                                    id="query-one",
                                    placeholder="e.g., A dog is running in the park.",
                                    style={
                                        "width": "100%",
                                        "minHeight": "86px",
                                        "resize": "vertical",
                                        "padding": "12px",
                                        "borderRadius": "12px",
                                        "border": f"1px solid {BORDER}",
                                        "backgroundColor": "rgba(255,255,255,0.04)",
                                        "color": TEXT,
                                        "outline": "none",
                                        "fontSize": "14px",
                                    },
                                ),

                                html.Div(style={"height": "12px"}),

                                html.Label("Sentence 2", style={"color": MUTED, "fontSize": "12px"}),
                                dcc.Textarea(
                                    id="query-two",
                                    placeholder="e.g., An animal is moving outdoors.",
                                    style={
                                        "width": "100%",
                                        "minHeight": "86px",
                                        "resize": "vertical",
                                        "padding": "12px",
                                        "borderRadius": "12px",
                                        "border": f"1px solid {BORDER}",
                                        "backgroundColor": "rgba(255,255,255,0.04)",
                                        "color": TEXT,
                                        "outline": "none",
                                        "fontSize": "14px",
                                    },
                                ),

                                html.Div(style={"height": "14px"}),

                                html.Div(
                                    [
                                        html.Button(
                                            "Analyze",
                                            id="search-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "10px 14px",
                                                "borderRadius": "12px",
                                                "border": "none",
                                                "background": "#2563eb",
                                                "color": "white",
                                                "fontWeight": "800",
                                                "cursor": "pointer",
                                            },
                                        ),
                                        html.Button(
                                            "Clear",
                                            id="clear-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "10px 14px",
                                                "borderRadius": "12px",
                                                "border": f"1px solid {BORDER}",
                                                "background": "transparent",
                                                "color": TEXT,
                                                "fontWeight": "700",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "10px"},
                                ),

                                html.Div(
                                    id="error-text",
                                    style={"marginTop": "12px", "color": "#fca5a5", "fontSize": "13px"},
                                ),
                            ],
                            style={
                                "backgroundColor": CARD_BG,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "18px",
                                "padding": "18px",
                                "boxShadow": "0 20px 60px rgba(0,0,0,0.35)",
                            },
                        ),

                        # Results
                        html.Div(
                            [
                                html.Div(
                                    "Result",
                                    style={"fontSize": "16px", "fontWeight": "800", "color": TEXT},
                                ),
                                html.Div(
                                    "Your model output appears below.",
                                    style={"fontSize": "12px", "color": MUTED, "marginTop": "4px"},
                                ),
                                html.Div(style={"height": "12px"}),
                                html.Div(
                                    id="search-results",
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "flex-start",
                                    },
                                ),
                            ],
                            style={
                                "marginTop": "14px",
                                "backgroundColor": "transparent",
                            },
                        ),
                    ],
                    style={
                        "maxWidth": "980px",
                        "margin": "0 auto",
                        "padding": "18px",
                    },
                )
            ],
        ),

        # Footer
        html.Div(
            [
                html.Div(
                    "A4 • Sentence similarity demo (custom BERT).",
                    style={"maxWidth": "980px", "margin": "0 auto", "padding": "18px", "color": MUTED, "fontSize": "12px"},
                )
            ],
            style={"borderTop": f"1px solid {BORDER}", "marginTop": "18px"},
        ),
    ],
    style={
        "minHeight": "100vh",
        "background": f"radial-gradient(1200px 600px at 20% 0%, rgba(37,99,235,0.25), transparent 60%), {APP_BG}",
        "fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial",
    },
)


# -----------------------------
# Callbacks
# -----------------------------

from dash import Input, Output, State, no_update, callback_context

@app.callback(
    Output("query-one", "value"),
    Output("query-two", "value"),
    Output("search-results", "children"),
    Output("error-text", "children"),
    Input("search-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("query-one", "value"),
    State("query-two", "value"),
    prevent_initial_call=True,
)
def handle_actions(n_analyze, n_clear, q1, q2):
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    # --- Clear pressed ---
    if triggered == "clear-button":
        return "", "", html.Div("Enter two sentences to see results.", style={"color": "#9ca3af"}), ""

    # --- Analyze pressed ---
    q1 = (q1 or "").strip()
    q2 = (q2 or "").strip()

    if not q1 or not q2:
        # don't wipe inputs; just show error
        return no_update, no_update, no_update, "Please fill both inputs."

    with torch.no_grad():
        score = calculate_similarity(model_bert, tokenizer, params["max_len"], q1, q2, device)

    label, color = classify_from_score(float(score))
    return no_update, no_update, result_card(label, color, float(score), q1, q2), ""


def clear_inputs(_):
    return "", "", html.Div("Enter two sentences to see results.", style={"color": "#9ca3af"}), ""

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
