from GoEmbeddings.po2vec_code import GOGraph, AnnotationLoader, train_po2vec, save_embeddings
import os

BASE_PATH = "/mnt/d/ML/Kaggle/CAFA6/cafa-6-protein-function-prediction/"

go_graph = GOGraph(os.path.join(BASE_PATH, 'Train', 'go-basic.obo'))
annotations = AnnotationLoader(os.path.join(BASE_PATH, 'Train', 'train_terms.tsv'))


# print(f"GO Graph has {len(go_graph.terms)} terms and {len(go_graph.edges)} edges.")

# Train model
model = train_po2vec(
    go_graph=go_graph,
    annotations=annotations,
    embedding_dim=512,  # or 512 for higher capacity
    epochs=10,
    batch_size=32,
    device='cuda',
    learning_rate=1e-3
)


# Save for later use
embeddings = save_embeddings(model, go_graph, 'go_embeddings.pkl')
