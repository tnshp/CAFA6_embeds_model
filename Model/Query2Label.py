import torch
import torch.nn as nn
import math


class MultiClassLinear(nn.Module):
    """Individual linear layer for each class (vectorized)."""
    
    def __init__(self, num_classes: int, in_features: int):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        
        # Weight: (num_classes, in_features)
        self.W = nn.Parameter(torch.Tensor(num_classes, in_features))
        # Bias: (num_classes,)
        self.b = nn.Parameter(torch.Tensor(num_classes))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.b, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_classes, in_features)
        Returns: (B, num_classes)
        """
        # Vectorized: einsum computes x @ W.T + b for each class
        logits = torch.einsum('bni,ni->bn', x, self.W) + self.b
        return logits


class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding for 1D sequences.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even
        pe[:, 1::2] = torch.cos(position * div_term)   # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        x = x + self.pe[:, :L]
        return self.dropout(x)


class Query2Label(nn.Module):
    """
    Generic Query2Label-style multi-label classifier.

    Expected input:
        backbone_features: (B, L, C_in)
            e.g. sequence or spatial features from a backbone (ESM, CNN, etc.)
    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        nheads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        hidden_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.pos_encoder = (
            PositionalEncoding(hidden_dim, dropout=dropout)
            if use_positional_encoding
            else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Learnable label embeddings as queries (one per class)
        self.label_queries = nn.Embedding(num_classes, hidden_dim)

        # Classification head: individual linear layer for each class
        self.classifier = MultiClassLinear(num_classes, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj.weight)
        # if self.in_proj.bias is not None:
        #     nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.label_queries.weight)
        # MultiClassLinear handles its own initialization

    def forward(
        self,
        backbone_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        backbone_features: (B, L, C_in)
        src_key_padding_mask: (B, L) with True for padded positions (optional)

        Returns:
            logits: (B, num_classes)  (use BCEWithLogitsLoss)
        """
        B, L, _ = backbone_features.shape

        # project to hidden_dim
        src = backbone_features

        # add positional encoding
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)  # (B, L, D)

        # encode features
        memory = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # (B, L, D)

        # label queries: (num_classes, D) → (B, num_classes, D)
        query = self.label_queries.weight.unsqueeze(0).expand(B, -1, -1)

        # no causal mask; all labels decoded in parallel
        tgt = query  # (B, num_classes, D)

        # decode: each label query attends to encoded features
        hs = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, num_classes, D)

        # classification per label (vectorized)
        logits = self.classifier(hs)  # (B, num_classes)

        return logits


# -------------------------