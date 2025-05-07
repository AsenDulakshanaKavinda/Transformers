import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.01):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

def test():
    # Dummy input (same shape as before)
    dummy_input = torch.rand(2, 5, 512)

    # FFN layer
    ffn = FeedForward(d_model=512)

    # Forward pass
    output = ffn(dummy_input)

    print("Output shape:", output.shape)  # Should be (2, 5, 512)
    print("Sample output (1st word of 1st batch):")
    print(output[0, 0, :10])


if __name__ == "__main__":
    test()














