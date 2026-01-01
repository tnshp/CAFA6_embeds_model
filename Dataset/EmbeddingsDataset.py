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

        sequence = self.data['sequences'][qseqid]

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
            'sequence'   : sequence,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': row['terms_true']
        }



def collate_tokenize(batch, tokenizer, device=None, dtype=torch.float32):
    """Custom collate function to handle variable-length data.
    
    Args:
        batch: List of samples from the dataset
        tokenizer: Tokenizer to apply to features
        device: Device to move tensors to (cuda or cpu)
        dtype: Target dtype for tensors (torch.float32, torch.float16, or torch.bfloat16)
    """
    sequences = [item['sequence'] for item in batch]
    # sequences = sequences.to(dtype=dtype, device=device) if device else sequences.to(dtype=dtype)
    enc_input = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc_input = {k: v.to(device) for k, v in enc_input.items()}
    
    go_embed = torch.stack([torch.from_numpy(item['go_embed']) for item in batch])
    go_embed = go_embed.to(dtype=dtype, device=device) if device else go_embed.to(dtype=dtype)
    
    label = torch.stack([torch.from_numpy(item['label']) for item in batch])
    label = label.to(dtype=dtype, device=device) if device else label.to(dtype=dtype)
    
    return {
        'entryID': [item['entryID'] for item in batch],
        'enc_input': enc_input,
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
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        if self.stream is not None:
            # CUDA prefetching
            return self._cuda_iter()
        else:
            # CPU fallback - no prefetching needed
            return iter(self.dataloader)
    
    def _cuda_iter(self):
        """Iterator with CUDA stream prefetching."""
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
                break
                    
            yield batch
    
    def _to_device(self, batch):
        """Move batch to device (already moved in collate_fn, but ensure it's there)."""
        if isinstance(batch, dict):
            # Batch is already on device from collate_fn, just return it
            return batch
        return batch
    
    def __len__(self):
        return len(self.dataloader)