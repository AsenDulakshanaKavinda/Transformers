import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # 5000x512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #256
        pe[:, 0::2] = torch.sin(position * div_term) # 5000x512
        pe[:, 1::2] = torch.cos(position * div_term) # 5000x512
        pe = pe.unsqueeze(0) # 1x5000x512
        self.register_buffer('pe', pe) # Register it as a buffer, meaning it’s not a trainable weight, but it stays with the model

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x



def test():
    batch_size = 2
    seq_len = 5000
    d_model = 512

    # Random input (normally this would be word embeddings)
    dummy_input = torch.zeros(batch_size, seq_len, d_model)

    # Create positional encoding module
    pos_encoder = PositionalEncoding(d_model=d_model)

    # Apply positional encoding
    output = pos_encoder(dummy_input)

    print("Output shape:", output.shape)  # Should be (2, 5000, 512)
    print("First sentence, first word embedding (after encoding):")
    print(output[0, 0, :10])  # Print first 10 values of the first position



if __name__ == "__main__":
    test()











