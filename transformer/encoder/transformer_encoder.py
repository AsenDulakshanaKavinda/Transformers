import torch
import torch.nn as nn

from positional_encoder import PositionalEncoding
from encoder_layer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_encoder(x))
        for layer in self.layers:
            x = layer(x, mask)

        return x
    

def test():
    # Dummy input: batch of 2 sentences, 10 tokens, each token has 512-dim embedding
    dummy_input = torch.rand(2, 10, 512)

    # Initialize encoder
    encoder = TransformerEncoder()

    # Forward pass
    output = encoder(dummy_input)

    # Output
    print("Output shape:", output.shape)  # Expected: (2, 10, 512)
    print("First token of first sequence:")
    print(output[0, 0, :10])  # Show first 10 values


if __name__ == "__main__":
    test()





















