"""
Phase 7 — Neural Collaborative Filtering (NCF) Model

Architecture:
  Input : user_idx, item_idx
  Embed : user_embedding(32), item_embedding(32)
  Concat: [user_emb || item_emb] → 64-dim vector
  FC    : 64 → 32 → 16 → 1
  Output: raw logit (apply sigmoid externally for probabilities)
"""
import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model.

    Learns non-linear user-item interaction patterns via:
      1. Embedding layers for users and items
      2. Concatenation of embeddings
      3. Multi-layer perceptron (MLP) for scoring
      4. Raw logit output (use BCEWithLogitsLoss for training)
    """
    def __init__(self, num_users, num_items, embed_dim=32, hidden_layers=[64, 32, 16]):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # MLP layers: input is concat of user + item embeddings = 2 * embed_dim
        mlp_input_dim = embed_dim * 2
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            mlp_input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer: single neuron — returns raw logit
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_idx, item_idx):
        """
        Forward pass — returns raw logits.

        Args:
            user_idx: LongTensor of user indices [batch_size]
            item_idx: LongTensor of item indices [batch_size]

        Returns:
            Raw logit tensor [batch_size] (apply sigmoid for probabilities)
        """
        # Step 1: Get embeddings
        user_emb = self.user_embedding(user_idx)   # [batch, embed_dim]
        item_emb = self.item_embedding(item_idx)   # [batch, embed_dim]

        # Step 2: Concatenate
        concat = torch.cat([user_emb, item_emb], dim=-1)  # [batch, embed_dim * 2]

        # Step 3: Pass through MLP
        mlp_out = self.mlp(concat)  # [batch, 16]

        # Step 4: Output raw logit
        logit = self.output_layer(mlp_out)  # [batch, 1]

        return logit.squeeze(-1)  # [batch]


if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 7 — NCF Model Test")
    print("=" * 60)

    num_users, num_items = 610, 9724
    model = NCF(num_users, num_items, embed_dim=32, hidden_layers=[64, 32, 16])
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    batch_size = 4
    u = torch.randint(0, num_users, (batch_size,))
    i = torch.randint(0, num_items, (batch_size,))
    model.eval()
    with torch.no_grad():
        logits = model(u, i)
        probs = torch.sigmoid(logits)
    print(f"Logits: {logits.tolist()}")
    print(f"Probs : {probs.tolist()}")
    print("=" * 60)
