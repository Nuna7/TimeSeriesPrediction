import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedEmbed(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(InvertedEmbed, self).__init__()
        self.embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0,2,1) # batch_size, embed_size, seq_len
        return self.dropout(self.embedding(x))

class AttentionLayer(nn.Module):
    def __init__(self, head_dim, d_model, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.head_dim = head_dim
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.final_proj = nn.Linear(d_model, d_model)

    def forward(self,q,k,v):
        B, H, D = q.shape

        queries = self.q_proj(q).view(B, H, self.num_heads, -1)
        keys = self.k_proj(k).view(B, H, self.num_heads, -1)
        values = self.v_proj(v).view(B, H, self.num_heads, -1)

        return self.final_proj(self.attention(queries, keys, values))

    def attention(self, q, k, v):
        B, D, NH, HD = q.shape
        scale = 1 / (HD ** 0.5)
        
        scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)

        atten_mask = torch.triu(torch.ones(D, D), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(atten_mask.to(scores.device), float('-inf'))

        attn_weights = torch.softmax(scores * scale, dim=-1)
 
        output = attn_weights @ v.permute(0, 2, 1, 3)

        output = output.permute(0, 2, 1, 3).contiguous().view(B, D, NH * HD)
        
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, head_dim, num_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = AttentionLayer(head_dim, d_model ,num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
        
class ITransformer(nn.Module):
    def __init__(self, seq_len, d_model=64, num_heads=4, dropout=0.1):
        super(ITransformer, self).__init__()
        self.seq_len = seq_len
        self.head_dim = d_model // num_heads
        self.embedding_layer = InvertedEmbed(seq_len, d_model)
        self.encoder = Encoder(d_model, self.head_dim, num_heads, dropout)
        self.final_projection = nn.Linear(d_model, seq_len)

    def forward(self, x):
        """
        x : Batch size, seq len, embed_size
        """
        x = self.embedding_layer(x)
        x = self.encoder(x)
        return self.final_projection(x).permute(0,2,1)