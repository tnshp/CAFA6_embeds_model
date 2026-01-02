import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EmbeddingsDataset(Dataset):
    """Dataset that tokenizes sequences and caches GO embeddings as tensors."""
    def __init__(self, 
                 data, 
                 tokenizer=None,
                 max_sequence_length=1024,
                 max_go_embeds = 256,  
                 oversample_indices=None
                ):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_go_embeds = max_go_embeds
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(self.data['seq_2_terms'])))
        
        # Pre-cache GO embeddings as tensors (significant speedup)
        print("Pre-caching GO embeddings as tensors...")
        self.go_embeds_tensor = {}
        for term, embed in self.data['go_embeds'].items():
            self.go_embeds_tensor[term] = torch.from_numpy(np.array(embed, dtype=np.float32))
        
        # Create a mask embedding tensor
        mask_shape = next(iter(self.data['go_embeds'].values())).shape
        self.mask_embed = torch.zeros(mask_shape, dtype=torch.float32)
        print(f"Cached {len(self.go_embeds_tensor)} GO embeddings as tensors.")


    def __len__(self):
        return len(self.oversample_indices)         

    def __getitem__(self, idx):
        sample_idx = self.oversample_indices[idx]

        row = self.data['seq_2_terms'].iloc[sample_idx]
        qseqid = row['qseqid']

        sequence = self.data['sequences'][qseqid]
        
        # Tokenize sequence in worker (not in collate_fn)
        if self.tokenizer is not None:
            enc_input = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                padding=False,  # Will pad in collate_fn
                truncation=True, 
                max_length=self.max_sequence_length
            )
            # Remove batch dimension added by return_tensors="pt"
            enc_input = {k: v.squeeze(0) for k, v in enc_input.items()}
        else:
            enc_input = None

        true_terms_set = set(row['terms_true'])
        predicted_terms = row['terms_predicted']
        
        # Use pre-cached tensor GO embeddings (much faster)
        valid_terms = predicted_terms
        go_embeds = torch.stack([self.go_embeds_tensor.get(term, self.mask_embed) for term in valid_terms])
        label = torch.tensor([term in true_terms_set for term in valid_terms], dtype=torch.float32)
        
        return {
            'entryID'   : qseqid,
            'enc_input' : enc_input,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': row['terms_true']
        }



def collate_tokenize(batch, dtype=torch.float32):
    """Custom collate function to handle variable-length tokenized data.
    Tensors stay on CPU; device transfer handled by PrefetchLoader.
    
    Args:
        batch: List of samples from the dataset (already tokenized)
        dtype: Target dtype for tensors (torch.float32, torch.float16, or torch.bfloat16)
    """
    # Pad tokenized inputs to same length in batch
    from torch.nn.utils.rnn import pad_sequence
    
    # Extract and pad input_ids and attention_mask
    input_ids = [item['enc_input']['input_ids'] for item in batch]
    attention_mask = [item['enc_input']['attention_mask'] for item in batch]
    
    # Pad sequences (keep on CPU)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    enc_input = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded
    }
    
    # Stack GO embeddings and labels (already tensors, keep on CPU)
    go_embed = torch.stack([item['go_embed'] for item in batch])
    go_embed = go_embed.to(dtype=dtype)
    
    label = torch.stack([item['label'] for item in batch])
    label = label.to(dtype=dtype)
    
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
        """Move batch to device."""
        if isinstance(batch, dict):
            result = {}
            for k, v in batch.items():
                if k == 'enc_input' and isinstance(v, dict):
                    # Handle nested dict for enc_input
                    result[k] = {ek: ev.to(self.device, non_blocking=True) if isinstance(ev, torch.Tensor) else ev 
                                 for ek, ev in v.items()}
                elif isinstance(v, torch.Tensor):
                    result[k] = v.to(self.device, non_blocking=True)
                else:
                    result[k] = v
            return result
        return batch
    
    def __len__(self):
        return len(self.dataloader)