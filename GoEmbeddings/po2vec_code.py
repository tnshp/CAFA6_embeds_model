
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import obonet
import numpy as np
from collections import defaultdict
import pickle
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Part 1: Load and Parse GO OBO File
# ============================================================================

class GOGraph:
    """
    Manages GO DAG from OBO file with efficient path computation.
    """
    def __init__(self, obo_path: str):
        logger.info(f"Loading GO from {obo_path}...")
        self.graph = obonet.read_obo(obo_path)
        
        # Create mappings
        self.id2term = {id_: data for id_, data in self.graph.nodes(data=True)}
        self.go_ids = list(self.graph.nodes())
        self.n_terms = len(self.go_ids)
        self.id2idx = {go_id: idx for idx, go_id in enumerate(self.go_ids)}
        
        logger.info(f"Loaded {self.n_terms} GO terms")
        
    def get_definition(self, go_id: str) -> str:
        """Get text definition of GO term"""
        if go_id in self.id2term:
            def_str = self.id2term[go_id].get('def', '')
            return def_str.strip('"')
        return ""
    
    def get_term_name(self, go_id: str) -> str:
        """Get name of GO term"""
        if go_id in self.id2term:
            return self.id2term[go_id].get('name', '')
        return ""
    
    def get_namespace(self, go_id: str) -> str:
        """Get namespace: biological_process, molecular_function, cellular_component"""
        if go_id in self.id2term:
            return self.id2term[go_id].get('namespace', 'unknown')
        return "unknown"
    
    def get_direct_neighbors(self, go_id: str) -> Set[str]:
        """Get direct parents and children (distance 1)"""
        if go_id not in self.graph:
            return set()
        
        neighbors = set()
        
        try:
            for parent in self.graph.successors(go_id):
                neighbors.add(parent)
        except:
            pass
        
        try:
            for child in self.graph.predecessors(go_id):
                neighbors.add(child)
        except:
            pass
        
        return neighbors
    
    def get_siblings(self, go_id: str) -> Set[str]:
        """Get sibling terms (share a parent)"""
        if go_id not in self.graph:
            return set()
        
        siblings = set()
        
        try:
            parents = list(self.graph.successors(go_id))
            for parent in parents:
                try:
                    for sibling in self.graph.predecessors(parent):
                        if sibling != go_id:
                            siblings.add(sibling)
                except:
                    pass
        except:
            pass
        
        return siblings

# ============================================================================
# Part 2: Load Protein-GO Annotations
# ============================================================================

class AnnotationLoader:
    """
    Load protein-GO term pairs from TSV file.
    Expected format: protein_id\tgo_term\n
    """
    def __init__(self, tsv_path: str):
        logger.info(f"Loading annotations from {tsv_path}...")
        self.protein_go_pairs = []
        self.protein_terms = defaultdict(set)
        self.term_proteins = defaultdict(set)
        
        with open(tsv_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    protein_id, go_term = parts[0], parts[1]
                    self.protein_go_pairs.append((protein_id, go_term))
                    self.protein_terms[protein_id].add(go_term)
                    self.term_proteins[go_term].add(protein_id)
        
        logger.info(f"Loaded {len(self.protein_go_pairs)} protein-GO annotations")
        logger.info(f"Unique proteins: {len(self.protein_terms)}")
        logger.info(f"Annotated GO terms: {len(self.term_proteins)}")
    
    def get_cooccurring_terms(self, go_id: str, min_cooccurrence: int = 1) -> Set[str]:
        """Get GO terms that co-occur with given term in same proteins"""
        if go_id not in self.term_proteins:
            return set()
        
        proteins = self.term_proteins[go_id]
        cooccurring = defaultdict(int)
        
        for protein in proteins:
            for term in self.protein_terms[protein]:
                if term != go_id:
                    cooccurring[term] += 1
        
        return {term for term, count in cooccurring.items() 
                if count >= min_cooccurrence}

# ============================================================================
# Part 3: Custom Collate Function with Device Handling
# ============================================================================

def po2vec_collate_fn(batch):
    """
    Custom collate function for variable-length positive/negative pairs.
    Returns lists that will be converted to tensors with correct device in forward().
    """
    anchors, positives, negatives = zip(*batch)
    anchors = torch.tensor(anchors, dtype=torch.long)
    return anchors, list(positives), list(negatives)

# ============================================================================
# Part 4: Contrastive Learning Dataset
# ============================================================================

class PO2VecDataset(Dataset):
    """
    Prepares positive and negative pairs for contrastive learning.
    """
    def __init__(self, 
                 go_graph: GOGraph,
                 annotations: AnnotationLoader,
                 n_negatives: int = 32,
                 use_siblings: bool = True,
                 min_cooccurrence: int = 1):
        
        self.go_graph = go_graph
        self.annotations = annotations
        self.n_negatives = n_negatives
        self.use_siblings = use_siblings
        
        self.annotated_ids = list(annotations.term_proteins.keys())
        self.n_terms = len(self.annotated_ids)
        self.id2idx = {go_id: idx for idx, go_id in enumerate(self.annotated_ids)}
        
        logger.info(f"Dataset size: {self.n_terms} annotated GO terms")
        
        self.positive_cache = {}
        self._build_positive_pairs(min_cooccurrence)
    
    def _build_positive_pairs(self, min_cooccurrence: int):
        """Build positive pairs using FAST direct neighbors and siblings"""
        logger.info("Building positive pairs (FAST: direct neighbors + siblings)...")
        
        for i, go_id in enumerate(self.annotated_ids):
            if (i + 1) % 5000 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.annotated_ids)} terms")
            
            positives = set()
            
            # Co-occurring terms
            cooccurring = self.annotations.get_cooccurring_terms(
                go_id, min_cooccurrence=min_cooccurrence
            )
            positives.update(cooccurring)
            
            # Direct neighbors
            neighbors = self.go_graph.get_direct_neighbors(go_id)
            positives.update(neighbors)
            
            # Siblings
            if self.use_siblings:
                siblings = self.go_graph.get_siblings(go_id)
                positives.update(siblings)
            
            # Keep only annotated positives
            valid_positives = [
                gid for gid in positives 
                if gid in self.id2idx
            ]
            
            self.positive_cache[go_id] = valid_positives
        
        logger.info("Positive pairs built")
    
    def __len__(self):
        return len(self.annotated_ids)
    
    def __getitem__(self, idx: int) -> Tuple[int, List[int], List[int]]:
        """
        Returns lists of integers (not tensors) for flexible device handling.
        """
        anchor_id = self.annotated_ids[idx]
        anchor_idx = self.id2idx[anchor_id]
        
        # Get positive samples
        positive_ids = self.positive_cache.get(anchor_id, [])
        if not positive_ids:
            positive_ids = [anchor_id]
        
        positive_indices = [self.id2idx[gid] for gid in positive_ids]
        
        # Sample negative examples
        positive_set = set(positive_indices)
        negative_pool = set(range(self.n_terms)) - positive_set
        
        if negative_pool:
            negative_indices = list(np.random.choice(
                list(negative_pool),
                size=min(self.n_negatives, len(negative_pool)),
                replace=False
            ))
        else:
            negative_indices = []
        
        return anchor_idx, positive_indices, negative_indices

# ============================================================================
# Part 5: PO2Vec Model with Device-Aware Forward
# ============================================================================

class PO2Vec(nn.Module):
    """
    PO2Vec Model with proper device handling.
    """
    def __init__(self, 
                 n_terms: int,
                 embedding_dim: int = 256,
                 temperature: float = 0.1):
        super().__init__()
        
        self.n_terms = n_terms
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        self.embeddings = nn.Embedding(n_terms, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, anchor_idx: torch.Tensor,
                positive_indices: List[List[int]],
                negative_indices: List[List[int]],
                device: str) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss with explicit device handling.
        
        FIX: Convert indices to tensors with device parameter inside forward()
        """
        batch_size = anchor_idx.shape[0]
        anchor_emb = self.embeddings(anchor_idx)
        
        loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            pos_list = positive_indices[i]
            neg_list = negative_indices[i]
            
            if len(neg_list) == 0:
                continue
            
            # CRITICAL FIX: Create tensors with device parameter
            pos_idx = torch.tensor(pos_list, dtype=torch.long, device=device)
            neg_idx = torch.tensor(neg_list, dtype=torch.long, device=device)
            
            anchor = anchor_emb[i:i+1]
            positives = self.embeddings(pos_idx)
            negatives = self.embeddings(neg_idx)
            
            pos_sim = F.cosine_similarity(anchor, positives) / self.temperature
            neg_sim = F.cosine_similarity(anchor, negatives) / self.temperature
            
            logits = torch.cat([pos_sim, neg_sim], dim=0)
            labels = torch.zeros_like(logits)
            labels[:len(pos_sim)] = 1.0
            
            sample_loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            loss += sample_loss
            valid_samples += 1
        
        if valid_samples > 0:
            return loss / valid_samples
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def get_embeddings(self) -> torch.Tensor:
        """Return all learned embeddings"""
        return self.embeddings.weight.detach().cpu()

# ============================================================================
# Part 6: Training Function
# ============================================================================

def train_po2vec(go_graph: GOGraph,
                 annotations: AnnotationLoader,
                 embedding_dim: int = 256,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 1e-3,
                 use_siblings: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_checkpoint: Optional[str] = None) -> PO2Vec:
    """
    Train PO2Vec model with fixed device handling.
    """
    logger.info(f"Training on {device}")
    
    dataset = PO2VecDataset(
        go_graph=go_graph,
        annotations=annotations,
        n_negatives=32,
        use_siblings=use_siblings
    )
    
    # FIX: Use custom collate function + num_workers=0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=po2vec_collate_fn,
        num_workers=0
    )
    
    model = PO2Vec(
        n_terms=go_graph.n_terms,
        embedding_dim=embedding_dim,
        temperature=0.1
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for batch_idx, (anchors, positives, negatives) in enumerate(pbar):
            anchors = anchors.to(device)
            
            optimizer.zero_grad()
            
            # FIX: Pass device to forward method
            loss = model(anchors, positives, negatives, device=device)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        
        avg_loss = total_loss / max(n_batches, 1)
        
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if save_checkpoint and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_checkpoint)
            logger.info(f"  → Saved checkpoint (loss improved to {best_loss:.4f})")
    
    logger.info("Training complete!")
    return model

# ============================================================================
# Part 7: Save and Evaluate Embeddings
# ============================================================================

def save_embeddings(model: PO2Vec,
                   go_graph: GOGraph,
                   output_path: str):
    """Save learned GO term embeddings to pickle file"""
    embeddings_tensor = model.get_embeddings()
    embeddings_np = embeddings_tensor.numpy()
    
    go_embeddings = {
        go_graph.go_ids[idx]: embeddings_np[idx]
        for idx in range(len(go_graph.go_ids))
    }
    
    output = {
        'embeddings': go_embeddings,
        'go_ids': go_graph.go_ids,
        'embedding_dim': model.embedding_dim,
        'n_terms': len(go_graph.go_ids)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    logger.info(f"Saved {len(go_embeddings)} embeddings to {output_path}")
    return go_embeddings

def compute_semantic_similarity(emb1: np.ndarray, 
                               emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))

# ============================================================================
# Part 8: Main Usage
# ============================================================================

if __name__ == "__main__":
    import os
    
    BASE_PATH = "/mnt/d/ML/Kaggle/CAFA6/cafa-6-protein-function-prediction"
    OBO_FILE = os.path.join(BASE_PATH, "Train", "go-basic.obo")
    TRAIN_TSV = os.path.join(BASE_PATH, "Train", "train_terms.tsv")
    OUTPUT_EMBEDDINGS = "go_embeddings.pkl"
    CHECKPOINT_PATH = "po2vec_checkpoint.pt"
    
    go_graph = GOGraph(OBO_FILE)
    annotations = AnnotationLoader(TRAIN_TSV)
    
    model = train_po2vec(
        go_graph=go_graph,
        annotations=annotations,
        embedding_dim=256,
        batch_size=32,
        epochs=100,
        learning_rate=1e-3,
        use_siblings=True,
        save_checkpoint=CHECKPOINT_PATH
    )
    
    embeddings = save_embeddings(model, go_graph, OUTPUT_EMBEDDINGS)
    
    if len(go_graph.go_ids) >= 2:
        go1, go2 = go_graph.go_ids[0], go_graph.go_ids[1]
        sim = compute_semantic_similarity(
            embeddings[go1],
            embeddings[go2]
        )
        logger.info(f"Similarity {go1} vs {go2}: {sim:.4f}")