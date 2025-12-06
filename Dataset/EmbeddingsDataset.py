# Tokenized dataset + collate that runs embeddings through the tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TokenizedEmbeddingsDataset(Dataset):
    """Dataset that yields raw embeddings; tokenization is done in collate_fn for batching."""
    def __init__(self, data, oversample_indices=None):
        self.data = data
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(self.data['embeds'])))

    def __len__(self):
        return len(self.oversample_indices)

    def __getitem__(self, idx):
        sample_idx = self.oversample_indices[idx]
        return {
            'entryID': self.data['entries'][sample_idx],
            'embed': self.data['embeds'][sample_idx],
            'label': self.data['labels'][sample_idx]
        }


def collate_tokenize(batch, tokenizer, device=None, dtype=torch.float32):
    """Batch raw embeds, move to device, run tokenizer once for the batch.

    Returns a dict with keys: 'entryID' (list), 'tokens' (Tensor: batch,N,d), 'label' (Tensor)
    """
    embeds = [item['embed'] for item in batch]
    # stack into (batch_size, D)
    embeds = np.stack(embeds)
    embeds = torch.tensor(embeds, dtype=dtype)
    if device is not None:
        embeds = embeds.to(device)

    # ensure tokenizer is on the requested device
    try:
        tokenizer_device = next(tokenizer.parameters()).device
    except StopIteration:
        # tokenizer may have no parameters; attempt P_buffer or default to cpu
        tokenizer_device = getattr(tokenizer, 'P_buffer', torch.tensor(0)).device if hasattr(tokenizer, 'P_buffer') else torch.device('cpu')

    if device is not None and tokenizer_device != device:
        try:
            tokenizer = tokenizer.to(device)
        except Exception:
            pass

    with torch.no_grad():
        tokens = tokenizer(embeds)  # expected (batch_size, N, d)

    # Fix swapped axes if tokenizer produced (N, batch_size, d)
    try:
        if hasattr(tokenizer, 'P_buffer') and isinstance(tokenizer.P_buffer, torch.Tensor):
            N = tokenizer.P_buffer.shape[0]
            if tokens.dim() == 3 and tokens.shape[0] == N and tokens.shape[1] != N:
                tokens = tokens.permute(1, 0, 2).contiguous()
    except Exception:
        pass

    # Convert labels to numpy array first to avoid slow list->tensor warning
    labels_np = np.array([item['label'] for item in batch])
    labels = torch.tensor(labels_np, dtype=torch.long)

    entryIDs = [item['entryID'] for item in batch]

    return {'entryID': entryIDs, 'tokens': tokens, 'label': labels}


class PrefetchLoader:
    """DataLoader wrapper that asynchronously preloads next batch to CUDA.

    Use like: prefetch_loader = PrefetchLoader(loader, device)
    for batch in prefetch_loader:
        # batch already moved to device (if CUDA)

    Notes:
    - Works only when `device` is CUDA; otherwise yields batches unchanged.
    - Requires that collate_fn returns torch tensors (tokens/label).
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            self._preload()
        else:
            self.next_batch = None
        return self

    def _preload(self):
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        # move tensors to device asynchronously
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                for k, v in list(self.next_batch.items()):
                    if isinstance(v, torch.Tensor):
                        # non_blocking requires pinned memory on CPU side
                        # If tensor already on CUDA, .cuda() will be a no-op on same device
                        self.next_batch[k] = v.cuda(non_blocking=True)

    def __next__(self):
        if self.device.type == 'cuda':
            if self.next_batch is None:
                raise StopIteration
            # wait for copy stream
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            # preload next
            self._preload()
            return batch
        else:
            # CPU path: just yield from underlying loader
            return next(self.loader_iter)
