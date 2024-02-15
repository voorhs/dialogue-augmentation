from torch import nn
import torch.nn.functional as F


class CLSPooling(nn.Module):
    """just take first token from each sequence in batch"""
    def forward(self, last_hidden_state, **kwargs):
        """
        last_hidden_state: (B,T,H)
        return: (B,H)
        """
        return last_hidden_state[:, 0, :]


class AveragePooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask, **kwargs):
        """
        last_hidden_state: (B,T,H)
        attention_mask: (B,T), mask where 0 indicates tokens not to attend
        return: (B,H)
        """
        hidden_states = last_hidden_state * attention_mask[..., None]
        return hidden_states.sum(dim=1) / attention_mask.sum(dim=1)


class SelfAttentionPooling(nn.Module):
    """
    output = Attention(K,Q,V), where
        K = inputs shape of (B,T,H)
        Q = trainable parameter shape of (H,)
        V = inputs
    
    without any normalization (l1, l2, and even division by sqrt(t))
    """

    def __init__(self, input_dim):
        super().__init__()
        
        self.Q = nn.Linear(input_dim, 1)
        
    def forward(self, last_hidden_state, **kwargs):
        """
        last_hidden_state: (B, T, H)
        return: (B, H)
        """

        att_scores = self.Q(last_hidden_state).squeeze(-1)              # (B,T)
        att_probs = F.softmax(att_scores)                           # (B,T)
        res = (last_hidden_state * att_probs[..., None]).sum(dim=1)     # (B,H)

        return res
