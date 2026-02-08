import torch
import torch.nn as nn
from Encoder_Layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        atten_type,
        device,              # kept for compatibility with your constructor calls
        max_length=100
    ):
        super().__init__()

        # You can keep these attributes for logging/debugging,
        # but we won't rely on self.device for tensor placement in forward.
        self.device = device
        self.atten_type = atten_type

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, atten_type, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # IMPORTANT: scale should move automatically with model.to(device)
        # and must NOT be pinned to the training device.
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(hid_dim))))

    def forward(self, src, src_mask):
        """
        src:      [batch size, src len]
        src_mask: [batch size, 1, 1, src len]
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # Create positional indices on the SAME device as src (CPU/GPU safe)
        pos = (
            torch.arange(0, src_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )  # [batch_size, src_len]

        # Embeddings + positional embeddings
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )  # [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
