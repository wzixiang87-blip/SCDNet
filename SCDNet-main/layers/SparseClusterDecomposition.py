import torch
import torch.nn as nn
import torch.nn.functional as F


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class ConstrainedSparseClusterDecomposition(nn.Module):
    def __init__(self,
                 d_model=128,
                 n_clusters=64,
                 top_k=4,
                 temperature=2.0,
                 seq_len=None,
                 pred_len=None):
        super().__init__()

        self.n_clusters = n_clusters
        self.d_model = d_model
        self.top_k = top_k

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_temperature = temperature

        self.dictionary = nn.Parameter(torch.randn(n_clusters, d_model))
        nn.init.orthogonal_(self.dictionary)

    def forward(self, x):
        B, N, D = x.shape

        # =========================================================
        # Step 1: 相似度
        # =========================================================
        scores = torch.matmul(x, self.dictionary.t())  # [B, N, K]

        # =========================================================
        # Step 2: Long-horizon temperature
        # =========================================================
        if self.seq_len is not None and self.pred_len is not None:
            temp = self.base_temperature * (1.0 + self.pred_len / self.seq_len)
        else:
            temp = self.base_temperature


        q = F.softmax(scores / temp, dim=-1)  # [B, N, K]
        q_flat = q.reshape(-1, self.n_clusters)

        with torch.no_grad():
            p_flat = target_distribution(q_flat)

        kl_loss = F.kl_div(q_flat.log(), p_flat, reduction='batchmean')


        gram = torch.matmul(self.dictionary, self.dictionary.t())
        identity = torch.eye(self.n_clusters, device=x.device)
        ortho_loss = F.mse_loss(gram, identity)

        if self.seq_len is not None and self.pred_len is not None:
            aux_loss = kl_loss * (self.seq_len / self.pred_len) + 0.1 * ortho_loss
        else:
            aux_loss = kl_loss + ortho_loss


        topk_values, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(-1, topk_indices, 1.0)

        scores_hard = scores.masked_fill(hard_mask == 0, float('-inf'))
        weights = F.softmax(scores_hard / temp, dim=-1)  # [B, N, K]


        dict_expand = self.dictionary.view(1, 1, self.n_clusters, D).expand(B, N, -1, -1)
        weights_expand = weights.unsqueeze(-1)  # [B, N, K, 1]

        x_common = torch.sum(weights_expand * dict_expand, dim=2)  # [B, N, D]


        x_residual = x - x_common

        return x_common, x_residual, aux_loss