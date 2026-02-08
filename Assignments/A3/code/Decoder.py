import torch
import torch.nn as nn
from Mutihead_Attention import MultiHeadAttentionLayer
from Feed_Forward import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, atten_type, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)

        self.self_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, atten_type, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, atten_type, device)
        self.feedforward       = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg      = [batch size, trg len, hid dim]
        # enc_src  = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, atten_type, device, max_length=100):
        super().__init__()

        # keep for compatibility, but DON'T rely on it in forward for tensor placement
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, atten_type, device)
            for _ in range(n_layers)
        ])

        self.fc_out  = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # IMPORTANT: don't pin scale to training device; let it move with model.to(...)
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(hid_dim))))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg      = [batch size, trg len]
        # enc_src  = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # Create positions on SAME device as trg (CPU/GPU safe)
        pos = (
            torch.arange(0, trg_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        # trg = [batch size, trg len, hid dim]

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        return output, attention
