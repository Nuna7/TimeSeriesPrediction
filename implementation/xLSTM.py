import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import logsigmoid

class CausalConv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        padding = (kernel_size - 1) * dilation
        super(CausalConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, **kwargs)

    def forward(self, input):
        result = super(CausalConv, self).forward(input)
        if self.padding[0] != 0:
            return result[:, :, :-self.padding[0]]
        return result

class MultiHeadLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(MultiHeadLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        B, S, NH, DH = x.shape
        gn_in = x.reshape(B * S, NH * DH)  # (B * S, NH * DH)

        out = F.group_norm(gn_in,num_groups=NH)
        
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out
        
    
class LinearHeadwiseExpand(nn.Module):
    def __init__(self,in_features,num_heads,bias=True):
        super(LinearHeadwiseExpand, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.bias = bias

        self.out_features = in_features
        self.out_features_per_head = self.out_features // num_heads
        self.in_features_per_head = in_features // num_heads

        self.weight = nn.Parameter(torch.zeros((num_heads, self.out_features_per_head, self.in_features_per_head)))

        if bias:
            self.bias = nn.Parameter(torch.zeros((num_heads, self.out_features_per_head)))
            nn.init.zeros_(self.bias.data)

        nn.init.normal_(self.weight.data, mean=0.0, std=(2 / 5 / self.weight.shape[-1]) ** 0.5)
            
    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x

class sLSTMCell(nn.Module):
    num_gates = 4
    
    def __init__(self, hidden_size, num_heads, num_states):
        super(sLSTMCell, self).__init__()
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_states = num_states
        self.hidden_size = hidden_size
        
        self.recurrent_kernel = torch.empty((num_heads, self.head_dim, self.num_gates, self.head_dim))
        self.bias = torch.empty(num_heads, self.num_gates, self.head_dim)
        
    def reset_parameters(self):
        for h in range(self.num_heads):
            for i in range(self.num_gates):
                nn.init.uniform_(
                    self.recurrent_kernel[h, :, i, :],
                    -1.0 / (self.hidden_size) ** 0.5,
                    1.0 / (self.hidden_size) ** 0.5
                )
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel)

        for h in range(self.num_heads):
            for i in range(self.num_gates):
                nn.init.uniform_(
                    self.bias[h, i],
                    -1 / (self.hidden_size) ** 0.5,
                    1 / (self.hidden_size) ** 0.5
                )
        self.bias = nn.Parameter(self.bias)


    def _zero_state(self, input):
        batch_dim = input.shape[1]
        state = torch.zeros((self.num_states, batch_dim, self.hidden_size))
        return state

    def _get_state(self, input, state=None):
        if state is None:
            state = self._zero_state(input)
        else:
            assert state.shape == (
                self.num_states,
                input.shape[1],
                self.hidden_size,
            )
        return state

    def _get_final_state(self, all_states):
        return all_states[:, -1]

    def forward(self, input, state=None):
        input = out.permute(1,0,2)
        states = self._get_state(input, state)
        
        all_states = self.slstm_forward(input, states)
        state = self._get_final_state(all_states)
        
        output = all_states[0][1:] # y or first is the hidden state
        return output, state

    def slstm_forward(self, x, states):  
        recurrent_kernel = self.recurrent_kernel.permute(0,2,3,1).reshape(self.num_heads,
                                                                              self.head_dim,
                                                                              self.num_heads * self.head_dim)
        
        num_states = states.shape[0]
        sequence_dim = x.shape[0]
        num_gates_r = self.num_gates
        hidden_dim = recurrent_kernel.shape[1] * recurrent_kernel.shape[0]
        batch_dim = x.shape[1]

        states_all = torch.zeros([num_states, sequence_dim + 1, batch_dim, hidden_dim])
        states_all[:, 0] = states # first input of all batches

        for i, Wx_t in enumerate(x.unbind(dim=0)):
            Ry = (
                # (batch, num_heads, 1, head_dim) @ (1 (broadcast for all batch), num_heads, head_dim, hidden_size)
                # -> batch, num_heads, 1, hidden_size) : dot product on last dim of first tensor and last second dim of second tensor
                states[0].reshape(batch_dim, self.num_heads, 1, -1).matmul(
                    recurrent_kernel.transpose(1, 2).reshape(1, self.num_heads, self.head_dim, self.num_gates * self.head_dim)
                )
                .reshape(batch_dim, self.num_heads, self.num_gates, -1)
                .transpose(1, 2)# Batch, num_gates, num_heads, head_dim (For reconstruction)
                .reshape(batch_dim, -1) # Batch, num_gates * num_heads * head_dim
            )

            states = self.lstm_forward(Wx_t, Ry, self.bias, states)
            states_all[:, i + 1] = states

        return states_all

    def lstm_forward(self, Wx, Ry, b, states):
        """
        hidden_state :  Represents the hidden state from the previous time step.
        cell_state :  Represents the cell state from the previous time step.
        n :  Represents the normalizer state from the previous time step.
        m :  Represents the stabilizer state from the previous time step
        """
        raw = Wx + Ry + b.reshape(1,-1)
        hidden_state, cell_state, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
        iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
        
        logfplusm = m + logsigmoid(fraw)
        if torch.all(n == 0.0):
            mnew = iraw
        else:
            mnew = torch.max(iraw, logfplusm)
            
        ogate = torch.sigmoid(oraw)
        igate = torch.exp(iraw - mnew)
        fgate = torch.exp(logfplusm - mnew)
        
        cell_state_new = fgate * cell_state + igate * torch.tanh(zraw)
        nnew = fgate * hidden_state + igate
        hidden_state_new = ogate * cell_state_new / nnew
    
        return torch.stack((hidden_state_new, cell_state_new, nnew, mnew), dim=0)
        

class sLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size=64 ,num_heads=4, num_states=4, pre_conv=True, dropout=0.2):
        super(sLSTM, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.pre_conv = pre_conv
        self.head_dim = hidden_size // num_heads

        if pre_conv:
            self.conv = CausalConv(self.embed_size, self.embed_size, 4)
            self.swish = nn.SiLU()

        self.forget_gate = LinearHeadwiseExpand(embed_size, num_heads, bias=False)
        self.input_gate = LinearHeadwiseExpand(embed_size, num_heads, bias=False)
        self.output_gate = LinearHeadwiseExpand(embed_size, num_heads, bias=False)
        self.z_gate = LinearHeadwiseExpand(embed_size, num_heads, bias=False)
        self.slstm_cell = sLSTMCell(hidden_size, self.num_heads, num_states)
        self.dropout = nn.Dropout(dropout)

        self.group_norm = MultiHeadLayerNorm(normalized_shape=embed_size)
        self.reset_parameters()
        self.lstm_state = None

    def reset_parameters(self):
        self.slstm_cell.reset_parameters()
        nn.init.normal_(self.group_norm.weight)
        nn.init.normal_(self.forget_gate.weight)
        nn.init.normal_(self.forget_gate.weight)
        nn.init.normal_(self.z_gate.weight)
        nn.init.normal_(self.output_gate.weight)

    def forward(self,x):
        B, S, _ = x.shape
        
        if self.pre_conv:
            x_conv = self.conv(x.permute(0,2,1)).permute(0,2,1)
            x_conv = self.swish(x.permute(0,2,1)).permute(0,2,1)
        else:
            x_conv = x

        i, f, z, o = (
            self.forget_gate(x_conv),
            self.input_gate(x_conv),
            self.z_gate(x),
            self.output_gate(x),
        )
        
        sLSTM_input = torch.cat([i, f, z, o], dim=-1)
        y, self.lstm_state = self.slstm_cell.forward(sLSTM_input,self.lstm_state)

        y = y.view(y.size(0), y.size(1), self.num_heads, self.head_dim)
        y = self.dropout(y)
        out = self.group_norm(y).transpose(1, 2).view(B, S, -1)

        return sLSTM_input

class mLSTMCell(nn.Module):
    def __init__(self, context_length, embed_size, num_heads):
        super(mLSTMCell, self).__init__()
        self.context_length = context_length
        self.embed_size = embed_size
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * embed_size, num_heads)
        self.fgate = nn.Linear(3 * embed_size, num_heads)

        self.causal_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))

        self.outnorm = MultiHeadLayerNorm(normalized_shape=embed_size)

    def forward(self, q, k, v):
        B, S, _ = q.shape
        assert self.context_length <= B
        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        h_state = self.parallel_lstm_forward(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def parallel_lstm_forward(self, queries, keys, values, igate_preact, fgate_preact, lower_triangular_matrix, stabilize_rowwise=True, eps=1e-6):

        B, NH, S, DH = queries.shape

        #Construction of log D of equation 74
        #-------
        log_fgates = torch.nn.functional.logsigmoid(fgate_preact) # Take log for stability

        # Append zero in order to satisfy diagonal element all being 0 at equation 71
        log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1)),
            torch.cumsum(log_fgates, dim=-2), # At specific S, will contain cumulative sum of all previsou S (Log gives us sum)
        ],dim=-2,
        )  # (B, NH, S+1, 1) 
         
        # We will make the lower triangular matrix where each entries are cumulative sum of all above rows and its sigmoid value
        rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1) 
        _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1) 

        # All entries above diagonal become -inifinity
        log_fg_matrix = torch.where(lower_triangular_matrix, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)
        #-------
        
        log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)

        if stabilize_rowwise:
            max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
        else:
            max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
            # (B, NH, 1, 1)
        
        log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
        D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)
    
        keys_scaled = keys / (DH**0.5)
    
        qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
        C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
        normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
        # (B, NH, S, S)
        C_matrix_normalized = C_matrix / (normalizer + eps)
    
        # retrieved values
        h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)
    
        return h_tilde_state

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fgate.weight)
        torch.nn.init.normal_(self.fgate.bias)
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class mLSTM(nn.Module):
    def __init__(self, embed_size ,hidden_size, context_length=256 ,num_heads=4, dropout=0.1):
        super(mLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = dropout

        self.qkv_proj_blocksize = 4

        self.proj_up = nn.Linear(embed_size, 2 * embed_size, bias=False)
        self.num_proj_heads = embed_size // self.qkv_proj_blocksize

        self.q_proj = LinearHeadwiseExpand(embed_size,num_heads,bias=False)
        self.k_proj = LinearHeadwiseExpand(embed_size,num_heads,bias=False)
        self.v_proj = LinearHeadwiseExpand(embed_size,num_heads,bias=False)

        self.conv = CausalConv(embed_size, embed_size, kernel_size=4)
        self.conv_act = nn.SiLU()

        self.mlstm_cell = mLSTMCell(context_length, embed_size ,num_heads)

        self.ogate_act = nn.SiLU()
        self.skip = nn.Parameter(torch.ones(embed_size, requires_grad=True))

        self.proj_down = nn.Linear(
            in_features=embed_size,
            out_features=embed_size,
            bias=True,
        )
        self.dropout = nn.Dropout(self.dropout)
        self.reset_parameters()

    def forward(self, x):
        B, S, _ = x.shape

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, split_size_or_sections=self.embed_size, dim=-1)

        x_mlstm_conv = self.conv(x_mlstm.permute(0,2,1)).permute(0,2,1)
        x_mlstm_conv_act = self.conv_act(x_mlstm_conv)
        
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.skip * x_mlstm_conv_act)

        h_state = h_tilde_state_skip * self.ogate_act(z)

        y = self.dropout(self.proj_down(h_state))
        
        return y

    def reset_parameters(self):
        nn.init.normal_(self.proj_up.weight)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
    
        nn.init.normal_(self.proj_down.weight)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.skip)
        self.mlstm_cell.reset_parameters()
        