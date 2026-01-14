import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EmbeddingsDataset(Dataset):
    """Dataset that yields raw PLM and BLM embeddings; tokenization and projection done in collate_fn."""
    def __init__(self, 
                 data, 
                 max_go_embeds = 256,  
                 oversample_indices=None
                ):
        
        self.data = data
        self.max_go_embeds = max_go_embeds
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(self.data['seq_2_terms'])))
        self.mask_embed = np.zeros(next(iter(self.data['go_embeds'].values())).shape, dtype=np.float32)
        
        # Get dimensions for PLM and BLM embeddings
        self.plm_dim = self.data['plm_embeds'][next(iter(self.data['plm_embeds']))].shape[0]
        self.blm_dim = self.data['pmid_2_embed'][next(iter(self.data['pmid_2_embed']))].shape[0]
        
        print(f"PLM dim: {self.plm_dim}, BLM dim: {self.blm_dim}")

    def __len__(self):
        return len(self.oversample_indices)         

    def __getitem__(self, idx):
        sample_idx = self.oversample_indices[idx]

        row = self.data['seq_2_terms'].iloc[sample_idx]
        qseqid = row['qseqid']

        plm_embed = self.data['plm_embeds'][qseqid]

        true_terms_set = set(row['terms_true'])
        predicted_terms = row['terms_predicted']
        
        # Filter terms that have embeddings (should be all of them after padding)
        valid_terms = predicted_terms
        # Vectorized operations using list comprehensions
        go_embeds = np.array([self.data['go_embeds'].get(term, self.mask_embed) for term in valid_terms])
        label = np.array([term in true_terms_set for term in valid_terms], dtype=np.float32)
        
        # Get BLM embeddings from PMIDs
        pmid_list = list(self.data['prot_2_pmid'].get(qseqid, []))
        
        # Skip None embeddings and collect valid ones
        valid_blm_embeds = [self.data['pmid_2_embed'].get(pmid) for pmid in pmid_list 
                           if self.data['pmid_2_embed'].get(pmid) is not None]
        
        # Create blm_embeds array - if no valid embeddings, create empty array with correct shape
        if len(valid_blm_embeds) > 0:
            blm_embeds = np.vstack(valid_blm_embeds)
        else:
            # Create empty array with shape [0, blm_dim]
            blm_embeds = np.zeros((0, self.blm_dim), dtype=np.float32)
        
        return {
            'entryID'   : qseqid,
            'plm_embed' : plm_embed,
            'blm_embeds': blm_embeds,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': row['terms_true']
        }



def simple_collate(batch):
    """Simple collate that returns batch as-is without stacking."""
    return batch


def collate_with_raw_features(batch, device=None, dtype=torch.float32, num_blm_tokens=32):
    """
    Collate function that prepares raw PLM and BLM features for the model to project internally.
    This ensures projection layers are part of the model and get saved with checkpoints.
    
    Args:
        batch: List of samples from the dataset
        device: Device to move tensors to (cuda or cpu)
        dtype: Target dtype for tensors (torch.float32, torch.float16, or torch.bfloat16)
        num_blm_tokens: Number of BLM tokens to pad to (default: 32)
    
    Returns:
        Dictionary with:
            - entryID: List of entry IDs
            - plm_raw: Raw PLM embeddings [batch, plm_dim]
            - blm_raw: Raw BLM embeddings [batch, num_blm_tokens, blm_dim]
            - blm_mask: BLM attention mask [batch, num_blm_tokens] (True = valid)
            - go_embed: GO embeddings [batch, num_terms, go_embed_dim]
            - label: Labels [batch, num_terms]
            - predicted_terms: List of predicted terms
            - true_terms: List of true terms
    """
    batch_size = len(batch)
    
    # Stack PLM embeddings
    plm_raw = torch.stack([torch.from_numpy(item['plm_embed']) for item in batch])
    plm_raw = plm_raw.to(dtype=dtype)
    if device is not None:
        plm_raw = plm_raw.to(device)
    
    # Get BLM dimension from first item
    blm_dim = batch[0]['blm_embeds'].shape[1] if batch[0]['blm_embeds'].shape[0] > 0 else 0
    
    # If blm_dim is 0, get it from data
    if blm_dim == 0:
        # Find first sample with BLM embeddings
        for item in batch:
            if item['blm_embeds'].shape[0] > 0:
                blm_dim = item['blm_embeds'].shape[1]
                break
    
    # If still 0, use a default (this shouldn't happen in practice)
    if blm_dim == 0:
        blm_dim = 768  # Default dimension
    
    # Prepare padded BLM embeddings and mask
    blm_raw = torch.zeros(batch_size, num_blm_tokens, blm_dim, dtype=dtype)
    blm_mask = torch.zeros(batch_size, num_blm_tokens, dtype=torch.bool)
    
    if device is not None:
        blm_raw = blm_raw.to(device)
        blm_mask = blm_mask.to(device)
    
    for i, item in enumerate(batch):
        blm = torch.from_numpy(item['blm_embeds']).to(dtype=dtype)  # [num_tokens, blm_dim]
        
        # Cap at num_blm_tokens
        actual_tokens = min(blm.shape[0], num_blm_tokens)
        if actual_tokens > 0:
            blm = blm[:actual_tokens]
            if device is not None:
                blm = blm.to(device)
            
            # Place in padded tensor
            blm_raw[i, :actual_tokens] = blm
            blm_mask[i, :actual_tokens] = True
    
    # Process GO embeddings and labels
    go_embed = torch.stack([torch.from_numpy(item['go_embed']) for item in batch])
    go_embed = go_embed.to(dtype=dtype)
    
    label = torch.stack([torch.from_numpy(item['label']) for item in batch])
    label = label.to(dtype=dtype)
    
    if device is not None:
        go_embed = go_embed.to(device)
        label = label.to(device)
    
    return {
        'entryID': [item['entryID'] for item in batch],
        'plm_raw': plm_raw,
        'blm_raw': blm_raw,
        'blm_mask': blm_mask,
        'go_embed': go_embed,
        'label': label,
        'predicted_terms': [item['predicted_terms'] for item in batch],
        'true_terms': [item['true_terms'] for item in batch]
    }


def collate_with_blm_projection(batch, blm_projection_layer, tokenizer, device=None, dtype=torch.float32, 
                                 num_plm_tokens=32, num_blm_tokens=32):
    """
    Custom collate function that tokenizes PLM features to 32 tokens and 
    projects BLM features to 32 tokens, stacking them to get 64 total tokens.
    
    Args:
        batch: List of samples from the dataset
        blm_projection_layer: nn.Linear layer to project BLM features from their dim to model_dim (512)
        tokenizer: Tokenizer to apply to PLM features (projects to 32 tokens) - REQUIRED
        device: Device to move tensors to (cuda or cpu)
        dtype: Target dtype for tensors (torch.float32, torch.float16, or torch.bfloat16)
        num_plm_tokens: Number of PLM tokens (default: 32)
        num_blm_tokens: Number of BLM tokens (default: 32)
    
    Returns:
        Dictionary with:
            - entryID: List of entry IDs
            - features: Stacked PLM + BLM tokens [batch, 64, model_dim]
            - mask: Attention mask [batch, 64]
            - go_embed: GO embeddings [batch, num_terms, go_embed_dim]
            - label: Labels [batch, num_terms]
            - predicted_terms: List of predicted terms
            - true_terms: List of true terms
    """
    batch_size = len(batch)
    model_dim = blm_projection_layer.out_features
    
    # Process PLM embeddings - stack and convert to tensors
    plm_embeds = torch.stack([torch.from_numpy(item['plm_embed']) for item in batch])
    plm_embeds = plm_embeds.to(dtype=dtype)
    
    if device is not None:
        plm_embeds = plm_embeds.to(device)
    
    # Tokenize PLM embeddings to fixed number of tokens (32)
    plm_tokens = tokenizer(plm_embeds)  # [batch, 32, model_dim]
    
    # Process BLM embeddings - project to model_dim, cap at 32, and pad to 32
    blm_embeds_padded = torch.zeros(batch_size, num_blm_tokens, model_dim, dtype=dtype)
    blm_attention_mask = torch.zeros(batch_size, num_blm_tokens, dtype=torch.bool)
    
    if device is not None:
        blm_embeds_padded = blm_embeds_padded.to(device)
        blm_attention_mask = blm_attention_mask.to(device)
    
    for i, item in enumerate(batch):
        blm = torch.from_numpy(item['blm_embeds']).to(dtype=dtype)  # [num_tokens, blm_dim]
        
        # Cap at num_blm_tokens (32)
        actual_tokens = min(blm.shape[0], num_blm_tokens)
        if actual_tokens > 0:
            blm = blm[:actual_tokens]
            
            if device is not None:
                blm = blm.to(device)
            
            # Project to model_dim (512)
            with torch.no_grad():
                blm_projected = blm_projection_layer(blm)  # [actual_tokens, model_dim]
            
            # Place in padded tensor
            blm_embeds_padded[i, :actual_tokens] = blm_projected
            blm_attention_mask[i, :actual_tokens] = True
    
    # Stack PLM tokens (32) and BLM tokens (32) to get 64 tokens total
    features = torch.cat([plm_tokens, blm_embeds_padded], dim=1)  # [batch, 64, model_dim]
    
    # Create combined attention mask (PLM tokens are always valid, BLM may be padded)
    plm_attention_mask = torch.ones(batch_size, num_plm_tokens, dtype=torch.bool)
    if device is not None:
        plm_attention_mask = plm_attention_mask.to(device)
    
    mask = torch.cat([plm_attention_mask, blm_attention_mask], dim=1)  # [batch, 64]
    
    # Process GO embeddings and labels
    go_embed = torch.stack([torch.from_numpy(item['go_embed']) for item in batch])
    go_embed = go_embed.to(dtype=dtype)
    
    label = torch.stack([torch.from_numpy(item['label']) for item in batch])
    label = label.to(dtype=dtype)
    
    if device is not None:
        go_embed = go_embed.to(device)
        label = label.to(device)
    
    return {
        'entryID': [item['entryID'] for item in batch],
        'features': features,
        'mask': mask,
        'go_embed': go_embed,
        'label': label,
        'predicted_terms': [item['predicted_terms'] for item in batch],
        'true_terms': [item['true_terms'] for item in batch]
    }


class PrefetchLoaderWithBLM:
    """
    Prefetch loader that loads batches asynchronously to GPU for faster training.
    Handles PLM tokenization (32 tokens), BLM projection (32 tokens), stacking to 64 tokens.
    """
    def __init__(self, dataloader, device, blm_projection_layer, tokenizer, 
                 num_plm_tokens=32, num_blm_tokens=32, max_prefetch=1):
        self.dataloader = dataloader
        self.device = device
        self.blm_projection_layer = blm_projection_layer
        self.tokenizer = tokenizer
        self.num_plm_tokens = num_plm_tokens
        self.num_blm_tokens = num_blm_tokens
        self.max_prefetch = max_prefetch
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
        # Move projection layer to device
        self.blm_projection_layer = self.blm_projection_layer.to(device)
        
        # Move tokenizer to device
        self.tokenizer = self.tokenizer.to(device)
        
    def __iter__(self):
        if self.stream is not None:
            return self._cuda_iter()
        else:
            return self._cpu_iter()
    
    def _cpu_iter(self):
        """Iterator without prefetching for CPU."""
        for batch in self.dataloader:
            # Apply collate with projection and tokenization
            batch = collate_with_blm_projection(
                batch, 
                self.blm_projection_layer,
                tokenizer=self.tokenizer,
                device=self.device, 
                dtype=next(self.blm_projection_layer.parameters()).dtype,
                num_plm_tokens=self.num_plm_tokens,
                num_blm_tokens=self.num_blm_tokens
            )
            yield batch
    
    def _cuda_iter(self):
        """Iterator with CUDA stream prefetching."""
        loader_iter = iter(self.dataloader)
        
        # Preload first batch
        try:
            with torch.cuda.stream(self.stream):
                next_batch = next(loader_iter)
                next_batch = self._process_batch(next_batch)
        except StopIteration:
            return
        
        while True:
            # Wait for the prefetch stream to finish
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            
            # Record stream for tensors
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v.record_stream(torch.cuda.current_stream())
            
            # Start loading next batch in background
            try:
                with torch.cuda.stream(self.stream):
                    next_batch = next(loader_iter)
                    next_batch = self._process_batch(next_batch)
            except StopIteration:
                yield batch
                del batch
                break
                
            yield batch
            del batch
    
    def _process_batch(self, batch):
        """Process batch with PLM tokenization, BLM projection, and move to device."""
        return collate_with_blm_projection(
            batch,
            self.blm_projection_layer,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=next(self.blm_projection_layer.parameters()).dtype,
            num_plm_tokens=self.num_plm_tokens,
            num_blm_tokens=self.num_blm_tokens
        )
    
    def __len__(self):
        return len(self.dataloader)


class PrefetchLoaderWithRawFeatures:
    """
    Prefetch loader that loads batches asynchronously to GPU for faster training.
    Prepares raw PLM and BLM features for the model to project internally.
    This ensures projection layers are part of the model and saved with checkpoints.
    """
    def __init__(self, dataloader, device, num_blm_tokens=32, max_prefetch=1):
        self.dataloader = dataloader
        self.device = device
        self.num_blm_tokens = num_blm_tokens
        self.max_prefetch = max_prefetch
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        if self.stream is not None:
            return self._cuda_iter()
        else:
            return self._cpu_iter()
    
    def _cpu_iter(self):
        """Iterator without prefetching for CPU."""
        for batch in self.dataloader:
            # Apply collate with raw features
            batch = collate_with_raw_features(
                batch, 
                device=self.device, 
                dtype=torch.float32,
                num_blm_tokens=self.num_blm_tokens
            )
            yield batch
    
    def _cuda_iter(self):
        """Iterator with CUDA stream prefetching."""
        loader_iter = iter(self.dataloader)
        
        # Preload first batch
        try:
            with torch.cuda.stream(self.stream):
                next_batch = next(loader_iter)
                next_batch = self._process_batch(next_batch)
        except StopIteration:
            return
        
        while True:
            # Wait for the prefetch stream to finish
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            
            # Record stream for tensors
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v.record_stream(torch.cuda.current_stream())
            
            # Start loading next batch in background
            try:
                with torch.cuda.stream(self.stream):
                    next_batch = next(loader_iter)
                    next_batch = self._process_batch(next_batch)
            except StopIteration:
                yield batch
                del batch
                break
                
            yield batch
            del batch
    
    def _process_batch(self, batch):
        """Process batch with raw features and move to device."""
        return collate_with_raw_features(
            batch,
            device=self.device,
            dtype=torch.float32,
            num_blm_tokens=self.num_blm_tokens
        )
    
    def __len__(self):
        return len(self.dataloader)
