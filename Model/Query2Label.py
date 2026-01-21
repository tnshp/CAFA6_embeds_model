import torch
import torch.nn as nn
import math


class CrossAttentionOnlyDecoderLayer(nn.Module):
    """Transformer decoder layer with ONLY cross-attention (no self-attention)."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Cross-attention only (no self-attention)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        self.activation = nn.functional.relu
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        tgt: (B, T, D) - target (queries)
        memory: (B, S, D) - encoder output
        memory_key_padding_mask: (B, S) - True for positions to ignore
        """
        # Cross-attention only
        attn_out, _ = self.multihead_attn(
            tgt, memory, memory, 
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)
        
        # Feed-forward
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff_out)
        tgt = self.norm2(tgt)
        
        return tgt


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
        plm_dim: int = None,
        blm_dim: int = None,
        num_plm_tokens: int = 32,
        nheads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        num_modalities: int = 2,
        modal_idx: list = [0, 32],
        dropout: float = 0.1,
        use_shared_classifier: bool = False,
        decoder_cross_attention_only: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        hidden_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_plm_tokens = num_plm_tokens
        
        # Create PLM projection layers (one per token position)
        # Each token position gets its own linear projection layer
        if plm_dim is not None:
            self.plm_projections = nn.ModuleList([
                nn.Linear(plm_dim, hidden_dim) for _ in range(num_plm_tokens)
            ])
            # Initialize each PLM projection layer distinctly using orthogonal initialization
            for idx, proj_layer in enumerate(self.plm_projections):
                # Use orthogonal init with different gain for each layer to ensure distinctness
                nn.init.orthogonal_(proj_layer.weight, gain=1.0 + idx * 0.01)
                nn.init.constant_(proj_layer.bias, 0.0)
        else:
            self.plm_projections = None
        
        # Create single BLM projection layer (shared across all BLM tokens)
        if blm_dim is not None:
            self.blm_projection = nn.Linear(blm_dim, hidden_dim)
            nn.init.xavier_uniform_(self.blm_projection.weight)
            nn.init.constant_(self.blm_projection.bias, 0.0)
        else:
            self.blm_projection = None
        
        self.pos_encoder = (
            PositionalEncoding(hidden_dim, dropout=dropout)
        )

        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, 1, hidden_dim) * 0.02
        )  
        self.modal_idx = modal_idx

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

        if decoder_cross_attention_only:
            # Use cross-attention only decoder layer
            decoder_layer = CrossAttentionOnlyDecoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            # Use standard transformer decoder layer with both self and cross attention
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

        # Classification head: choose between shared or multi-class classifier
        self.use_shared_classifier = use_shared_classifier
        if use_shared_classifier:
            # Shared linear layer for all classes
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            # Individual linear layer for each class
            self.classifier = MultiClassLinear(num_classes, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize transformer layers with default PyTorch initialization
        # MultiClassLinear handles its own initialization
        pass

    def forward(
        self,
        query: torch.Tensor, 
        backbone_features: torch.Tensor,
        plm_features: torch.Tensor = None,
        blm_features: torch.Tensor = None,
        blm_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query: (B, num_classes, D) 
        backbone_features: (B, L, C_in) - if plm_features and blm_features are None, use this directly
        plm_features: (B, plm_dim) - raw PLM embeddings to be projected
        blm_features: (B, max_blm_tokens, blm_dim) - raw BLM embeddings to be projected
        blm_mask: (B, max_blm_tokens) - mask for valid BLM tokens (True = valid)
        src_key_padding_mask: (B, L) with True for padded positions (optional)

        Returns:
            logits: (B, num_classes)  (use BCEWithLogitsLoss)
        """
        B = query.size(0)
        
        # If raw PLM and BLM features are provided, project them
        if plm_features is not None and blm_features is not None:
            # Project PLM features: each token position gets its own projection
            plm_tokens = torch.stack([
                proj_layer(plm_features) for proj_layer in self.plm_projections
            ], dim=1)  # (B, num_plm_tokens, hidden_dim)
            
            # Project BLM features: shared projection for all BLM tokens
            # blm_features: (B, max_blm_tokens, blm_dim)
            B_blm, max_blm_tokens, blm_dim = blm_features.shape
            blm_flat = blm_features.reshape(-1, blm_dim)  # (B * max_blm_tokens, blm_dim)
            blm_projected = self.blm_projection(blm_flat)  # (B * max_blm_tokens, hidden_dim)
            blm_tokens = blm_projected.reshape(B_blm, max_blm_tokens, self.hidden_dim)  # (B, max_blm_tokens, hidden_dim)
            
            # Concatenate PLM and BLM tokens
            src = torch.cat([plm_tokens, blm_tokens], dim=1)  # (B, num_plm_tokens + max_blm_tokens, hidden_dim)
            
            # Create combined attention mask
            plm_mask = torch.ones(B, self.num_plm_tokens, dtype=torch.bool, device=plm_features.device)
            if blm_mask is None:
                # If no BLM mask provided, assume all BLM tokens are valid
                blm_mask = torch.ones(B, max_blm_tokens, dtype=torch.bool, device=blm_features.device)
            combined_mask = torch.cat([plm_mask, blm_mask], dim=1)  # (B, num_plm_tokens + max_blm_tokens)
            
            # Update src_key_padding_mask for transformer (True = ignore)
            src_key_padding_mask = ~combined_mask  # Invert: True = padded/ignore
        else:
            # Use pre-projected backbone_features directly
            src = backbone_features
            # src_key_padding_mask already provided (or None)
        
        L = src.size(1)

        #add modal encoding 
        for i in range(len(self.modal_idx)):
            if i == len(self.modal_idx) - 1:
                src[:, self.modal_idx[i]: , :] += self.modality_embeddings[i]
            else:
                src[:, self.modal_idx[i]: self.modal_idx[i+1], :] += self.modality_embeddings[i]

        # encode features
        memory = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # (B, L, D)

        # no causal mask; all labels decoded in parallel
        tgt = query  # (B, num_classes, D)

        # decode: each label query attends to encoded features
        hs = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, num_classes, D)

        # classification per label
        if self.use_shared_classifier:
            # Shared classifier: apply same linear layer to all class tokens
            logits = self.classifier(hs).squeeze(-1)  # (B, num_classes, 1) -> (B, num_classes)
        else:
            # Multi-class classifier: different linear layer per class (vectorized)
            logits = self.classifier(hs)  # (B, num_classes)

        return logits


# -------------------------