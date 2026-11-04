import torch.nn as nn


class MLPHead(nn.Module):
    """Map the final [CLS] representation to unnormalized class logits.

    This experiment uses GELU in the hidden layer. The original ViT's optional
    hidden classification layer used tanh; GELU is kept here as our chosen variant.
    Apply softmax outside the head when displaying probabilities.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
