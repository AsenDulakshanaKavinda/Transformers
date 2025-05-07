import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers to create Q, K, V
        self.q_linear = nn.Linear(d_model, d_model) # 512x512
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # final linear layer after all heads are concatenated
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.size()

        # give learnable param to QKV
        Q = self.q_linear(query) # 2x10x512 * 512x512 -> 2x10x512
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim) (2, 8, 10,512)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # 2, 8, 10, 10

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        output = output.transpose(1,2).contiguous().view(B, T, self.d_model)

        return self.out_linear(output)



def test():
    # Settings
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create dummy input: random embeddings for a batch of sequences
    dummy_input = torch.rand(batch_size, seq_len, d_model)

    # Initialize Multi-Head Attention
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    # Forward pass (self-attention: Q=K=V=dummy_input)
    output = mha(dummy_input, dummy_input, dummy_input)

    # Check output shape
    print("Output shape:", output.shape)  # Expected: (2, 5, 512)
    print("Sample output (1st word of 1st batch):")
    print(output[0, 0, :10])  # Show first 10 values of the first token



if __name__ == "__main__":
    test()

















