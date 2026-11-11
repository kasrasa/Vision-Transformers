import torch
import torch.nn as nn


class SingleAttentionHead(nn.Module):
    """Single self-attention head."""

    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        self.query = nn.Linear(hidden_dim, head_dim)
        self.key = nn.Linear(hidden_dim, head_dim)
        self.value = nn.Linear(hidden_dim, head_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_tokens, hidden_dim).
        """
        Q = self.query(x) # (batch_size, num_tokens, head_dim)
        K = self.key(x)   # (batch_size, num_tokens, head_dim)
        V = self.value(x) # (batch_size, num_tokens, head_dim)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return output

class MultiHeadSelfAttention(nn.Module):
    """Independent heads followed by an output projection.

    Choose hidden_dim divisible by num_heads.
    """

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.heads = nn.ModuleList([SingleAttentionHead(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_tokens, hidden_dim).
        """
        head_outputs = [head(x) for head in self.heads]
        concatenated_output = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated_output)
        return output

class TransformerEncoderLayer(nn.Module):
    """One pre-normalized encoder block, following the original ViT diagram.

    The final encoder normalization belongs after the full stack of these blocks.
    """

    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim)
        )
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_tokens, hidden_dim).
        """
        # Normalize the attention branch; the residual path keeps x unchanged.
        attention_output = self.self_attention(self.layernorm1(x))
        x = x + self.dropout(attention_output)

        # Normalize the MLP branch, then add its update to the residual stream.
        feedforward_output = self.feedforward(self.layernorm2(x))
        x = x + self.dropout(feedforward_output)

        return x

