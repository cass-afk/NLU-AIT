import torch
import torch.nn as nn
from Additive_Attention import AdditiveAttention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, atten_type, device):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.atten_type = atten_type

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # IMPORTANT: do NOT pin scale to the constructor device.
        # Register as a buffer so it moves automatically with model.to(...)
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(self.head_dim))))

        # Additive attention module
        self.additive_attention = AdditiveAttention(self.head_dim)

        # keep device attribute only for compatibility/debugging (not used for tensor placement)
        self.device = device

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key   = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q,K,V: [batch_size, len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q,K,V: [batch_size, n_heads, len, head_dim]

        # Attention scores
        if self.atten_type == "multiplicative":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        elif self.atten_type == "general":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        elif self.atten_type == "additive":
            energy = self.additive_attention(Q, K)
        else:
            raise ValueError("atten_type must be 'multiplicative', 'general', or 'additive'.")

        # Mask padding
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        # x: [batch_size, n_heads, query len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [batch_size, query len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
