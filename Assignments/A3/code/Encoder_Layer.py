import torch.nn as nn
from Mutihead_Attention import MultiHeadAttentionLayer
from Feed_Forward import PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, atten_type, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, atten_type, device
        )
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src      = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
