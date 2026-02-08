import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        # NOTE: storing full encoder/decoder objects inside params is not portable for torch.save
        # but we keep it for compatibility with your existing code.
        self.params = {
            "encoder": encoder,
            "decoder": decoder,
            "src_pad_idx": src_pad_idx,
            "trg_pad_idx": trg_pad_idx
        }

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        # keep mask on same device as src automatically
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        # IMPORTANT FIX: create on trg.device, not self.device
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=trg.device)
        ).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention

    @classmethod
    def load_model(cls, model_path, device):
        # CPU-safe load
        params, state = torch.load(model_path, map_location=device)

        # If params contains a device, override it
        if isinstance(params, dict):
            params["device"] = device

        model = cls(**params, device=device).to(device)
        model.load_state_dict(state)
        model.eval()
        return model
