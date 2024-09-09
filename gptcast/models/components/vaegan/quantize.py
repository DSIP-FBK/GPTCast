import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
# from collections import Counter


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, sane_index_shape=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.codebook_counter = Counter()

        self.sane_index_shape = sane_index_shape

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1) # this operation is not differentiable
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        # we detach the z_q so that the quantization does not affect the gradient backpropagation
        # we pretend that z_q carries the same gradient as z (straight-through estimator)
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # update codebook counter
        # self.codebook_counter.update(min_encoding_indices.cpu().numpy().astype(int))
        
        # compute perplexity, i.e. how many codes are used on average
        # perplexity = np.exp(np.mean([np.log2(len(self.codebook_counter))]))
        # # perplexity = exp(-1 * sum(p(x) * log(p(x))) = exp(-1 * sum(count(x) / total * log(count(x) / total)))
        # perplexity = np.exp(np.mean([np.log(self.codebook_counter[i]) for i in self.codebook_conter]))

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
            
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
