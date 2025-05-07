from multi_head_self_attention import MultiHeadSelfAttention
from feed_forward import FeedForward

import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # mha + add norm
        atten_output = self.mha(x, x, x, mask=None)
        x = self.norm1(x + self.dropout1(atten_output))

        # ff + add and norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


def test():

    # Dummy input
    dummy_input = torch.rand(2, 5, 512)

    # Create one encoder layer
    encoder_layer = EncoderLayer(d_model=512, num_heads=8)

    # Forward pass
    output = encoder_layer(dummy_input)

    print("Output shape:", output.shape)  # Should be (2, 5, 512)
    print("Sample output (1st word of 1st batch):")
    print(output[0, 0, :10])


if __name__ == "__main__":
    test()














