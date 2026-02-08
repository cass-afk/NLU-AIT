import os
import torch

# Import your model code (adjust names to match your project)
from Encoder import Encoder
from Decoder import Decoder, DecoderLayer
from S2S import Seq2SeqTransformer
from Encoder_Layer import EncoderLayer
from Feed_Forward import PositionwiseFeedforwardLayer
from Additive_Attention import AdditiveAttention
from Mutihead_Attention import MultiHeadAttentionLayer

# IMPORTANT: help pickle find classes that were saved from __main__ (notebook)
globals()["Encoder"] = Encoder
globals()["Decoder"] = Decoder
globals()["Seq2SeqTransformer"] = Seq2SeqTransformer
globals()["EncoderLayer"] = EncoderLayer
globals()["DecoderLayer"] = DecoderLayer
globals()["PositionwiseFeedforwardLayer"] = PositionwiseFeedforwardLayer
globals()["AdditiveAttention"] = AdditiveAttention
globals()["MultiHeadAttentionLayer"] = MultiHeadAttentionLayer

device = torch.device("cpu")

old_path = os.path.normpath(r"C:\\Users\\Lenovo\\Desktop\\AIT\\NLU\\Assignments\\A3\\NLP\\A3\\model\\multiplicative_Seq2SeqTransformer.pt")   # <-- your old GPU ckpt
new_path = os.path.normpath(r"C:\\Users\\Lenovo\\Desktop\\AIT\\NLU\\Assignments\\A3\\NLP\\A3\\model\\multiplicative_Seq2SeqTransformer.clean.pt")

# Load GPU checkpoint onto CPU
obj = torch.load(old_path, map_location=device)

# Your training saved as [params, state_dict]
if isinstance(obj, (list, tuple)) and len(obj) == 2:
    params, state = obj
else:
    raise ValueError(f"Unexpected checkpoint format: {type(obj)}")

# Force CPU inside params
if isinstance(params, dict):
    params["device"] = device

# Save portable checkpoint
torch.save({"params": params, "state_dict": state}, new_path)
print("✅ Saved clean checkpoint to:", new_path)
