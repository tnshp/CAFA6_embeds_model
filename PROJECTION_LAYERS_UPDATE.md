# Projection Layers Update - Model Architecture Change

## Problem
Previously, PLM (Protein Language Model) and BLM (Biomedical Literature Model) features were projected in the collate function using external tokenizer and projection layers. These layers were NOT part of the model state dict, causing issues when loading saved checkpoints:
- The projection layers weren't saved with the model
- When loading a model, new random projections were created
- This caused drastic metric drops when reproducing results

## Solution
Moved projection layers into the model itself:

### 1. **Query2Label Model** (`Model/Query2Label.py`)
Added projection layers as part of the model:

```python
def __init__(
    self,
    num_classes: int,
    in_dim: int,
    plm_dim: int = None,        # NEW: Input dimension of PLM features
    blm_dim: int = None,        # NEW: Input dimension of BLM features
    num_plm_tokens: int = 32,   # NEW: Number of PLM tokens to project
    ...
):
    # PLM projections: num_plm_tokens separate linear layers
    if plm_dim is not None:
        self.plm_projections = nn.ModuleList([
            nn.Linear(plm_dim, hidden_dim) for _ in range(num_plm_tokens)
        ])
        # Initialize each layer distinctly using orthogonal initialization
        for idx, proj_layer in enumerate(self.plm_projections):
            nn.init.orthogonal_(proj_layer.weight, gain=1.0 + idx * 0.01)
            nn.init.constant_(proj_layer.bias, 0.0)
    
    # BLM projection: single shared linear layer for all BLM tokens
    if blm_dim is not None:
        self.blm_projection = nn.Linear(blm_dim, hidden_dim)
        nn.init.xavier_uniform_(self.blm_projection.weight)
        nn.init.constant_(self.blm_projection.bias, 0.0)
```

### 2. **Forward Pass** supports both modes:

```python
def forward(
    self,
    query: torch.Tensor,
    backbone_features: torch.Tensor = None,  # Pre-projected features (legacy)
    plm_features: torch.Tensor = None,       # Raw PLM features (NEW)
    blm_features: torch.Tensor = None,       # Raw BLM features (NEW)
    blm_mask: torch.Tensor = None,           # BLM mask (NEW)
    src_key_padding_mask: torch.Tensor = None,
):
    # If raw features provided, project them internally
    if plm_features is not None and blm_features is not None:
        plm_tokens = torch.stack([
            proj_layer(plm_features) for proj_layer in self.plm_projections
        ], dim=1)
        
        blm_tokens = self.blm_projection(blm_features)
        src = torch.cat([plm_tokens, blm_tokens], dim=1)
    else:
        # Use pre-projected features (backward compatibility)
        src = backbone_features
```

### 3. **New Collate Function** (`Dataset/EmbeddingsDataset.py`)

Added `collate_with_raw_features()` that prepares raw PLM/BLM embeddings instead of projecting them:

```python
def collate_with_raw_features(batch, device=None, dtype=torch.float32, num_blm_tokens=32):
    """
    Returns:
        - plm_raw: Raw PLM embeddings [batch, plm_dim]
        - blm_raw: Raw BLM embeddings [batch, num_blm_tokens, blm_dim]
        - blm_mask: BLM attention mask [batch, num_blm_tokens]
        - go_embed: GO embeddings [batch, num_terms, go_embed_dim]
        - label: Labels [batch, num_terms]
    """
```

### 4. **New Prefetch Loader** (`Dataset/EmbeddingsDataset.py`)

Added `PrefetchLoaderWithRawFeatures` that works with raw features:

```python
class PrefetchLoaderWithRawFeatures:
    """
    Prefetch loader for raw features - no external projection layers.
    Model handles projections internally.
    """
```

### 5. **Training Script** (`train.py`)

Updated to support both modes via config flag `use_raw_features`:

```python
# New approach (default)
use_raw_features = training_configs.get('use_raw_features', True)

if use_raw_features:
    # Model handles projection internally
    model = Query2Label_pl(
        num_classes=num_classes,
        in_dim=token_d,
        plm_dim=plm_embedding_dim,  # Pass raw dimensions
        blm_dim=blm_embedding_dim,  # Pass raw dimensions
        num_plm_tokens=num_plm_tokens,
        ...
    )
    # Use raw features loader
    train_loader = PrefetchLoaderWithRawFeatures(...)
else:
    # Legacy mode: external tokenizer and projection
    model = Query2Label_pl(
        num_classes=num_classes,
        in_dim=token_d,
        plm_dim=None,  # No internal projection
        blm_dim=None,
        ...
    )
    # Use pre-projection loader
    train_loader = PrefetchLoaderWithBLM(...)
```

## Usage

### Training with new approach (recommended):
```json
{
    "training_configs": {
        "use_raw_features": true,
        ...
    }
}
```

### Loading saved models:
```python
# Models trained with use_raw_features=True will have projection layers in state dict
model = Query2Label_pl.load_from_checkpoint('checkpoint.ckpt')

# Projection layers are now part of the model:
# - model.model.plm_projections (ModuleList with num_plm_tokens layers)
# - model.model.blm_projection (Single Linear layer)
```

## Key Benefits

1. **Reproducibility**: Projection layers are saved with model checkpoints
2. **Consistency**: No mismatch between training and inference projections
3. **Simplicity**: All learnable parameters in one place
4. **Backward Compatibility**: Old mode still works via `use_raw_features=False`

## Initialization Details

### PLM Projections (distinct initialization):
- Each of the `num_plm_tokens` projection layers is initialized distinctly
- Uses orthogonal initialization with varying gain: `gain = 1.0 + idx * 0.01`
- Ensures each token position has unique learned transformations

### BLM Projection (shared initialization):
- Single projection layer shared across all BLM tokens
- Uses Xavier uniform initialization
- All BLM tokens use the same learned transformation

## Migration Guide

To migrate existing training configs:
1. Add `"use_raw_features": true` to training_configs
2. Remove any external tokenizer/projection setup
3. Retrain models - old checkpoints won't be compatible

For backward compatibility with old checkpoints:
1. Keep `"use_raw_features": false` (or omit it)
2. Continue using external tokenizer/projection layers
3. Gradual migration as models are retrained
