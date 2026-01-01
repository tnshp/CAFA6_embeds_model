import numpy as np
import torch
import torch.nn as nn

def nearly_orthogonal_vectors(N, K, tol=1e-2, max_tries=10, rng=None):
    """
    Generate K nearly orthogonal random unit vectors in R^N.

    Parameters
    ----------
    N : int
        Dimension of the space.
    K : int
        Number of vectors to generate.
    tol : float
        Maximum allowed absolute inner product between distinct vectors.
    max_tries : int
        Maximum attempts per vector before giving up.
    rng : np.random.Generator or None
        Random generator. If None, use default.

    Returns
    -------
    V : ndarray, shape (K, N)
        Approximate orthonormal set of vectors (rows).
    """
    if rng is None:
        rng = np.random.default_rng()

    V = []
    for i in range(K):
        for t in range(max_tries):
            # start with a random Gaussian vector
            v = rng.standard_normal(N)

            # Gram–Schmidt: remove components along previously accepted vectors
            for u in V:
                v = v - np.dot(v, u) * u

            # if the vector is nearly zero, restart
            nrm = np.linalg.norm(v)
            if nrm < 1e-10:
                continue
            v = v / nrm

            # check near-orthogonality to all previous
            if all(abs(np.dot(v, u)) <= tol for u in V):
                V.append(v)
                break
        else:
            raise RuntimeError(
                f"Could not find vector {i+1} with tol={tol} in {max_tries} tries"
            )

    return np.vstack(V)

def generate_orthogonal_vectors_qr(dimension, num_vectors=None):
    """
    Generates a set of orthogonal vectors using QR factorization.
    
    Args:
        dimension (int): The dimensionality of the vectors.
        num_vectors (int, optional): The number of vectors to generate. 
                                     Defaults to 'dimension' for a full basis.
    Returns:
        numpy.ndarray: A matrix where each column is an orthogonal vector.
    """
    if num_vectors is None:
        num_vectors = dimension
        
    # Generate a random matrix
    A = np.random.rand(dimension, num_vectors)
    
    # Perform QR factorization
    # The 'Q' matrix contains the orthogonal columns
    Q, R = np.linalg.qr(A)
    
    # Return the first 'num_vectors' columns of Q
    return Q[:, :num_vectors]

class EmbedTokenizer(nn.Module):
    def __init__(self, D, d, N, rng=None):  
        super(EmbedTokenizer, self).__init__()
        """
        Parameters
        ----------
        D : int
            Dimension of the embedding space.
        d : int
            Dimension of the token space.
        N : int
            Number of tokens.
        rng : np.random.Generator or None
            Random generator. If None, use default.
        """
        if rng is None:
            rng = np.random.default_rng()

        V = generate_orthogonal_vectors_qr(D)
        K = D
        # Build P as a single tensor of shape (N, d, D) and register as buffer so it's moved with .to()
        P_list = []
        for i in range(N):
            indices = np.arange(D)
            sampled_idx = np.random.choice(indices, size=d, replace=False)
            p = V[:, sampled_idx].T
            P_list.append(p)

        P_np = np.stack(P_list, axis=0).astype(np.float32)  # (N, d, D)
        P_tensor = torch.from_numpy(P_np)
        # register as parameter so it can be trained if needed
        self.P_buffer = nn.Parameter(P_tensor, requires_grad=True)

    def forward(self, x):

        """
        Parameters
        ----------
        x : Tensor, shape (D)
            Input embeddings.

        Returns
        -------
        tokens : Tensor, shape (batch_size, N, d)
            Token representations.
        """
        # x: (batch_size, D) or (D,) -> ensure batch
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Get P and move to correct dtype
        P = self.P_buffer.to(dtype=x.dtype)  # (N, d, D)

        # Vectorized matmul: (batch_size, D) @ (D, N*d) -> (batch_size, N*d) -> reshape
        D = x.shape[1]
        P_2d = P.permute(2, 0, 1).reshape(D, -1)  # (D, N*d)
        tokens = torch.matmul(x, P_2d)  # (batch_size, N*d)
        tokens = tokens.reshape(x.shape[0], P.shape[0], P.shape[1])  # (batch_size, N, d)

        if squeeze_output:
            tokens = tokens.squeeze(0)

        return tokens
