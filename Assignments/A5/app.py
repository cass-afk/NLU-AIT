import dash
from dash import dcc, html, Input, Output, State, callback_context
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Load model and tokenizer
# =========================
model_name = "cass-afk/a5"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 has no pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
model.eval()

def generate_response(prompt, max_new_tokens=120, temperature=0.8, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# =========================
# Dash app
# =========================
app = dash.Dash(__name__)
app.title = "DPO Chat"

# Simple theme tokens
COLORS = {
    "bg": "#0b1220",
    "card": "#111a2e",
    "card2": "#0f172a",
    "border": "rgba(255,255,255,0.10)",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "accent": "#60a5fa",
    "accent2": "#22c55e",
    "danger": "#ef4444",
}

def chat_bubble(text, who="user"):
    is_user = (who == "user")
    return html.Div(
        text,
        style={
            "maxWidth": "85%",
            "alignSelf": "flex-end" if is_user else "flex-start",
            "background": COLORS["accent"] if is_user else COLORS["card2"],
            "color": "white" if is_user else COLORS["text"],
            "padding": "10px 12px",
            "borderRadius": "14px",
            "lineHeight": "1.4",
            "whiteSpace": "pre-wrap",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.25)",
            "border": f"1px solid {COLORS['border']}",
        },
    )

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": f"radial-gradient(1000px 600px at 20% 0%, rgba(96,165,250,0.18), transparent),"
                      f"radial-gradient(900px 500px at 80% 0%, rgba(34,197,94,0.12), transparent),"
                      f"{COLORS['bg']}",
        "display": "flex",
        "justifyContent": "center",
        "padding": "24px 12px",
        "fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial",
    },
    children=[
        html.Div(
            style={
                "width": "min(960px, 100%)",
                "display": "grid",
                "gridTemplateColumns": "1fr 320px",
                "gap": "16px",
            },
            children=[
                # =========================
                # Main Chat Panel
                # =========================
                html.Div(
                    style={
                        "background": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "16px",
                        "boxShadow": "0 12px 40px rgba(0,0,0,0.35)",
                        "display": "flex",
                        "flexDirection": "column",
                        "overflow": "hidden",
                    },
                    children=[
                        # Header
                        html.Div(
                            style={
                                "padding": "16px 16px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "space-between",
                                "borderBottom": f"1px solid {COLORS['border']}",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            "DPO Chat",
                                            style={"fontSize": "18px", "fontWeight": 700, "color": COLORS["text"]},
                                        ),
                                        html.Div(
                                            "Chat with your fine-tuned model",
                                            style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "2px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="status-pill",
                                    style={
                                        "fontSize": "12px",
                                        "color": COLORS["muted"],
                                        "padding": "6px 10px",
                                        "borderRadius": "999px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "background": "rgba(255,255,255,0.03)",
                                    },
                                    children=f"Device: {DEVICE}",
                                ),
                            ],
                        ),

                        # Chat history
                        dcc.Loading(
                            type="dot",
                            children=html.Div(
                                id="chat-window",
                                style={
                                    "padding": "16px",
                                    "height": "60vh",
                                    "overflowY": "auto",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "10px",
                                    "background": "rgba(0,0,0,0.08)",
                                },
                            ),
                        ),

                        # Input bar
                        html.Div(
                            style={
                                "padding": "12px",
                                "borderTop": f"1px solid {COLORS['border']}",
                                "background": "rgba(0,0,0,0.12)",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "alignItems": "stretch"},
                                    children=[
                                        dcc.Textarea(
                                            id="user-input",
                                            placeholder="Type a message… (Shift+Enter for newline)",
                                            style={
                                                "flex": 1,
                                                "height": "54px",
                                                "borderRadius": "12px",
                                                "border": f"1px solid {COLORS['border']}",
                                                "padding": "12px",
                                                "background": "rgba(255,255,255,0.04)",
                                                "color": COLORS["text"],
                                                "resize": "none",
                                                "outline": "none",
                                                "fontSize": "14px",
                                            },
                                        ),
                                        html.Button(
                                            "Send",
                                            id="send-btn",
                                            n_clicks=0,
                                            style={
                                                "width": "110px",
                                                "borderRadius": "12px",
                                                "border": "none",
                                                "background": COLORS["accent"],
                                                "color": "white",
                                                "fontWeight": 700,
                                                "cursor": "pointer",
                                                "boxShadow": "0 10px 22px rgba(96,165,250,0.25)",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="error-text",
                                    style={"marginTop": "8px", "fontSize": "12px", "color": COLORS["danger"]},
                                ),
                            ],
                        ),
                    ],
                ),

                # =========================
                # Settings Panel
                # =========================
                html.Div(
                    style={
                        "background": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "16px",
                        "boxShadow": "0 12px 40px rgba(0,0,0,0.35)",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Div(
                            "Generation Settings",
                            style={
                                "padding": "16px",
                                "fontWeight": 700,
                                "color": COLORS["text"],
                                "borderBottom": f"1px solid {COLORS['border']}",
                            },
                        ),
                        html.Div(
                            style={"padding": "16px", "display": "flex", "flexDirection": "column", "gap": "14px"},
                            children=[
                                html.Div(
                                    children=[
                                        html.Div("Max new tokens", style={"color": COLORS["muted"], "fontSize": "12px"}),
                                        dcc.Slider(
                                            id="max-new-tokens",
                                            min=32,
                                            max=256,
                                            step=8,
                                            value=120,
                                            marks={32: "32", 120: "120", 256: "256"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div("Temperature", style={"color": COLORS["muted"], "fontSize": "12px"}),
                                        dcc.Slider(
                                            id="temperature",
                                            min=0.1,
                                            max=1.5,
                                            step=0.1,
                                            value=0.8,
                                            marks={0.1: "0.1", 0.8: "0.8", 1.5: "1.5"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div("Top-p", style={"color": COLORS["muted"], "fontSize": "12px"}),
                                        dcc.Slider(
                                            id="top-p",
                                            min=0.1,
                                            max=1.0,
                                            step=0.05,
                                            value=0.9,
                                            marks={0.1: "0.1", 0.9: "0.9", 1.0: "1.0"},
                                        ),
                                    ]
                                ),
                                html.Hr(style={"borderColor": COLORS["border"]}),
                                html.Div(
                                    style={"display": "flex", "gap": "10px"},
                                    children=[
                                        html.Button(
                                            "Clear chat",
                                            id="clear-btn",
                                            n_clicks=0,
                                            style={
                                                "flex": 1,
                                                "borderRadius": "12px",
                                                "border": f"1px solid {COLORS['border']}",
                                                "background": "rgba(255,255,255,0.04)",
                                                "color": COLORS["text"],
                                                "padding": "10px 12px",
                                                "cursor": "pointer",
                                            },
                                        ),
                                        html.Button(
                                            "Insert sample",
                                            id="sample-btn",
                                            n_clicks=0,
                                            style={
                                                "flex": 1,
                                                "borderRadius": "12px",
                                                "border": "none",
                                                "background": COLORS["accent2"],
                                                "color": "white",
                                                "padding": "10px 12px",
                                                "fontWeight": 700,
                                                "cursor": "pointer",
                                                "boxShadow": "0 10px 22px rgba(34,197,94,0.22)",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "marginTop": "6px",
                                        "fontSize": "12px",
                                        "color": COLORS["muted"],
                                        "lineHeight": "1.4",
                                    },
                                    children=[
                                        html.Div("Tips:"),
                                        html.Ul(
                                            style={"margin": "8px 0 0 16px"},
                                            children=[
                                                html.Li("Lower temperature for more consistent answers."),
                                                html.Li("Increase max new tokens for longer responses."),
                                                html.Li("If you hit OOM, lower max new tokens."),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # Stores
                dcc.Store(
                    id="chat-store",
                    data=[{"role": "assistant", "content": "Hi! Ask me anything."}],
                ),
            ],
        )
    ],
)

# =========================
# Callbacks
# =========================

@app.callback(
    Output("user-input", "value", allow_duplicate=True),
    Input("sample-btn", "n_clicks"),
    prevent_initial_call=True,
)
def insert_sample(n):
    return "Explain in simple terms how oil is refined into gasoline."

@app.callback(
    Output("chat-store", "data"),
    Output("user-input", "value"),
    Output("error-text", "children"),
    Input("send-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    State("user-input", "value"),
    State("chat-store", "data"),
    State("max-new-tokens", "value"),
    State("temperature", "value"),
    State("top-p", "value"),
)
def on_send_or_clear(send_clicks, clear_clicks, user_text, chat, max_new_tokens, temperature, top_p):
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if triggered == "clear-btn":
        return ([{"role": "assistant", "content": "Chat cleared. Ask me anything."}], "", "")

    if triggered == "send-btn":
        if not user_text or not user_text.strip():
            return (chat, user_text, "Please type a message before sending.")

        user_text = user_text.strip()

        # Append user message
        chat = chat + [{"role": "user", "content": user_text}]

        # Build a simple prompt from history (GPT-2 style)
        # You can improve this format if you trained with a specific chat template.
        prompt = ""
        for m in chat[-6:]:  # keep last few turns to limit context length
            if m["role"] == "user":
                prompt += f"User: {m['content']}\n"
            else:
                prompt += f"Assistant: {m['content']}\n"
        prompt += "Assistant:"

        try:
            answer = generate_response(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            # Try to strip the prompt echo if it appears
            if "Assistant:" in answer:
                answer = answer.split("Assistant:", 1)[-1].strip()
            chat = chat + [{"role": "assistant", "content": answer or "(No output)"}]
            return (chat, "", "")
        except RuntimeError as e:
            # Most common runtime error is OOM
            msg = str(e)
            if "out of memory" in msg.lower():
                return (chat, user_text, "CUDA OOM: try lowering max new tokens or temperature.")
            return (chat, user_text, f"Runtime error: {e}")
        except Exception as e:
            return (chat, user_text, f"Error: {e}")

    return (chat, user_text, "")

@app.callback(
    Output("chat-window", "children"),
    Input("chat-store", "data"),
)
def render_chat(chat):
    bubbles = []
    for m in chat:
        bubbles.append(chat_bubble(m["content"], who="user" if m["role"] == "user" else "assistant"))
    return bubbles

if __name__ == "__main__":
    app.run(debug=True)