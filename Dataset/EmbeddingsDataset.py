import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EmbeddingsDataset(Dataset):
    """Dataset that yields raw embeddings; tokenization is done in collate_fn for batching."""
    def __init__(self, 
                 data, 
                 max_go_embeds = 256,  
                 oversample_indices=None
                ):
        
        self.data = data
        self.max_go_embeds = max_go_embeds
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(self.data['seq_2_terms'])))
        self.mask_embed = np.zeros(next(iter(self.data['go_embeds'].values())).shape, dtype=np.float32)
        #ensure len of predicted go terms is less than max_go_embeds
        #self.data['seq_2_terms'] = self.data['seq_2_terms'][self


    def __len__(self):
        return len(self.oversample_indices)         

    def __getitem__(self, idx):
        sample_idx = self.oversample_indices[idx]

        row = self.data['seq_2_terms'].iloc[sample_idx]
        qseqid = row['qseqid']

        feature_embed = self.data['features_embeds'][qseqid]

        true_terms_set = set(row['terms_true'])
        predicted_terms = row['terms_predicted']
        
        # Filter terms that have embeddings (should be all of them after padding)
        # valid_terms = [term for term in predicted_terms if term in self.data['go_embeds']]
        valid_terms = predicted_terms
        # Vectorized operations using list comprehensions
        go_embeds = np.array([self.data['go_embeds'].get(term, self.mask_embed) for term in valid_terms])
        label = np.array([term in true_terms_set for term in valid_terms], dtype=np.float32)
        
        return {
            'entryID'   : qseqid,
            'feature'   : feature_embed,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': row['terms_true']
        }



def collate_tokenize(batch, tokenizer=None, device=None, dtype=torch.float32):
    """Custom collate function to handle variable-length data.
    
    Args:
        batch: List of samples from the dataset
        tokenizer: Tokenizer to apply to features (if None, tokenization happens later)
        device: Device to move tensors to (cuda or cpu) - DEPRECATED, use None for multi-worker support
        dtype: Target dtype for tensors (torch.float32, torch.float16, or torch.bfloat16)
    
    Note: For multi-worker DataLoader support (num_workers > 0), keep device=None and 
          move tensors to GPU after collation using PrefetchLoader or in training loop.
    """
    features = torch.stack([torch.from_numpy(item['feature']) for item in batch])
    features = features.to(dtype=dtype)
    
    # Only tokenize if tokenizer is provided and on CPU
    # For multi-worker support, tokenization should happen after collation
    if tokenizer is not None:
        # Move tokenizer to CPU if needed for worker compatibility
        if hasattr(tokenizer, 'P_buffer') and tokenizer.P_buffer.device.type != 'cpu':
            # Tokenizer is on GPU, which doesn't work with workers
            # Skip tokenization here - it will be done in PrefetchLoader
            pass
        else:
            features = tokenizer(features)
    
    go_embed = torch.stack([torch.from_numpy(item['go_embed']) for item in batch])
    go_embed = go_embed.to(dtype=dtype)
    
    label = torch.stack([torch.from_numpy(item['label']) for item in batch])
    label = label.to(dtype=dtype)
    
    # Don't move to device here - let PrefetchLoader handle it
    return {
        'entryID': [item['entryID'] for item in batch],
        'feature': features,
        'go_embed': go_embed,
        'label': label,
        'predicted_terms': [item['predicted_terms'] for item in batch],
        'true_terms': [item['true_terms'] for item in batch]
    }


class PrefetchLoader:
    """
    Prefetch loader that loads batches asynchronously to GPU for faster training.
    Overlaps data transfer with computation by loading the next batch while 
    the model processes the current batch.
    
    Also handles tokenization on GPU if tokenizer is provided.
    """
    def __init__(self, dataloader, device, tokenizer=None, max_prefetch=1):
        self.dataloader = dataloader
        self.device = device
        self.tokenizer = tokenizer
        self.max_prefetch = max_prefetch  # Limit prefetching to reduce memory usage
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        if self.stream is not None:
            # CUDA prefetching
            return self._cuda_iter()
        else:
            # CPU fallback - no prefetching needed
            return iter(self.dataloader)
    
    def _cuda_iter(self):
        """Iterator with CUDA stream prefetching and memory management."""
        loader_iter = iter(self.dataloader)
        
        # Preload first batch
        try:
            with torch.cuda.stream(self.stream):
                next_batch = next(loader_iter)
                next_batch = self._to_device(next_batch)
        except StopIteration:
            return
        
        while True:
            # Wait for the prefetch stream to finish
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            
            # Make sure tensors are ready before yielding
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v.record_stream(torch.cuda.current_stream())
            
            # Start loading next batch in background
            try:
                with torch.cuda.stream(self.stream):
                    next_batch = next(loader_iter)
                    next_batch = self._to_device(next_batch)
            except StopIteration:
                yield batch
                # Clean up
                del batch
                break
                
            yield batch
            # Free memory from previous batch
            del batch
    
    def _to_device(self, batch):
        """Move batch to device and apply tokenization if needed."""
        if isinstance(batch, dict):
            result = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(device=self.device, non_blocking=True)
                else:
                    result[k] = v
            
            # Apply tokenizer on GPU after moving to device
            if self.tokenizer is not None and 'feature' in result:
                result['feature'] = self.tokenizer(result['feature'])
            
            return result
        return batch
    
    def __len__(self):
        return len(self.dataloader)