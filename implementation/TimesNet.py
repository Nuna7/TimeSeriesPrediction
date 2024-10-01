import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=4):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, c_out ,embed_size, k=10, d_model=64, dff=16, num_kernels=6):
        super(TimesNet, self).__init__()
        self.embed_size = embed_size
        self.d_model = d_model
        self.num_kernels = num_kernels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = k

        self.conv = nn.Sequential(
            Inception_Block(d_model, dff, num_kernels=4),
            nn.GELU(),
            Inception_Block(dff, d_model, num_kernels=4)
        )

        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.embedding_layer = nn.Linear(embed_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        xf = torch.fft.rfft(x, dim=1)
        frequencies = abs(xf).mean(0).mean(-1)
        # This output Tensor with dim 1 = (number of frequency) which is the avg amplitude of each frequency across all variates 
        # The abs give us the amplitude by taking magnitude of complex numbers result in real number

        frequencies[0] = 0
        # sets the amplitude of the 0th frequency to 0, as the 0th frequency (DC component) usually represents the mean of the signal 
        # and may not be informative for period detection.
        _, top_list = torch.topk(frequencies, self.k)
        top_list = top_list.detach().cpu().numpy()
        all_periods = x.shape[1] // top_list

        top_amplitude = abs(xf).mean(-1)[:, top_list]

        result = []
        for i in range(self.k):
            period = all_periods[i]

            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(x.shape[0], length // period, period, x.shape[2]).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)

            out = out.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[2])
            result.append(out[:, :(self.seq_len + self.pred_len), :])

        result = torch.stack(result, dim=-1)
        top_amplitude = F.softmax(top_amplitude, dim=1)
        top_amplitude = top_amplitude.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[1], x.shape[2], 1)

        result = torch.sum(result * top_amplitude, -1)
        result = result + x
        return self.projection(result)