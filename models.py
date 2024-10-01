import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

from constants import (
    XLSTM_RAINFALL,
    XLSTM_WATER,
    XLSTM_WEATHER,
    ITRANSFORMER_RAINFALL,
    ITRANSFORMER_WATER,
    ITRANSFORMER_WEATHER,
    TIMESNET_RAINFALL,
    TIMESNET_WATER,
    TIMESNET_WEATHER
)

from iTransformer import iTransformer
from implementation.TimesNet import TimesNet

subdivision_to_idx = {
    'ANDAMAN & NICOBAR ISLANDS': 0,
    'ARUNACHAL PRADESH': 1,
    'ASSAM & MEGHALAYA': 2,
    'BIHAR': 3,
    'CHHATTISGARH': 4,
    'COASTAL ANDHRA PRADESH': 5,
    'COASTAL KARNATAKA': 6,
    'EAST MADHYA PRADESH': 7,
    'EAST RAJASTHAN': 8,
    'EAST UTTAR PRADESH': 9,
    'GANGETIC WEST BENGAL': 10,
    'GUJARAT REGION': 11,
    'HARYANA DELHI & CHANDIGARH': 12,
    'HIMACHAL PRADESH': 13,
    'JAMMU & KASHMIR': 14,
    'JHARKHAND': 15,
    'KERALA': 16,
    'KONKAN & GOA': 17,
    'LAKSHADWEEP': 18,
    'MADHYA MAHARASHTRA': 19,
    'MATATHWADA': 20,
    'NAGA MANI MIZO TRIPURA': 21,
    'NORTH INTERIOR KARNATAKA': 22,
    'ORISSA': 23,
    'PUNJAB': 24,
    'RAYALSEEMA': 25,
    'SAURASHTRA & KUTCH': 26,
    'SOUTH INTERIOR KARNATAKA': 27,
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': 28,
    'TAMIL NADU': 29,
    'TELANGANA': 30,
    'UTTARAKHAND': 31,
    'VIDARBHA': 32,
    'WEST MADHYA PRADESH': 33,
    'WEST RAJASTHAN': 34,
    'WEST UTTAR PRADESH': 35
}

cities_to_idx = {
    'Beijing' : 0,
    'California' : 1,
    'London' : 2,
    'Singapore' : 3,
    'Tokyo' : 4
}

# XLSTM
class XLSTM(nn.Module):
    def __init__(self, cfg, input_channel, embed, pred_len, type_, proj=True):
        super(XLSTM, self).__init__()
        if proj:
            self.proj = nn.Linear(input_channel, embed)
        else:
            self.proj = nn.Identity()

        cfg.embedding_dim = embed
        xlstm_stack = xLSTMBlockStack(cfg)
        self.xlstm = xlstm_stack.to("cpu")
        self.projection = nn.Linear(cfg.embedding_dim * cfg.context_length, pred_len)

        if type_ == "classification":
            self.last = nn.Sigmoid()
            self.loss_function = nn.BCELoss()
        else:
            self.last = nn.Identity()
            self.loss_function = nn.MSELoss()

    
    def forward(self, x):
        B, _, _ = x.shape
        x = self.proj(x)
        x = self.projection(self.xlstm(x).reshape(B, -1))
        return self.last(x)

cfg_weather = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=31,
    num_blocks=7,
    embedding_dim=256,
    slstm_at=[1],
)
    
xlstm_weather = XLSTM(cfg_weather, 11, 256, 3, type_="regression")
xlstm_weather.load_state_dict(torch.load(XLSTM_WEATHER))


cfg_water = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=8,
    num_blocks=7,
    embedding_dim=256,
    slstm_at=[1],
)

xlstm_water = XLSTM(cfg_water, 1, 256 , 1, type_="classification")
xlstm_water.load_state_dict(torch.load(XLSTM_WATER))


cfg_rainfall = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=5,
    num_blocks=7,
    embedding_dim=256,
    slstm_at=[1],
)

xlstm_rainfall = XLSTM(cfg_rainfall, 48, 256 , 12, type_="regression")
xlstm_rainfall.load_state_dict(torch.load(XLSTM_RAINFALL))

# INVERTED TRANSFORMER
class ITransformer(nn.Module):
    def __init__(self, cfg, input_channel, embed, pred_len, task, type_, proj=True):
        super(ITransformer, self).__init__()
        self.pred_len = pred_len
        if proj:
            self.proj = nn.Linear(input_channel, embed)
            cfg["num_variates"] = embed
        else:
            self.proj = nn.Identity()

        itransformer = iTransformer(**cfg)
        self.itransformer = itransformer.to("cpu")
        self.projection = nn.Linear(cfg['num_variates'], 1)
        self.type = type_

        if task == "classification":
            self.last = nn.Sigmoid()
            self.loss_function = nn.BCELoss()
        else:
            self.last = nn.Identity()
            self.loss_function = nn.MSELoss()

    
    def forward(self, x):
        x = self.proj(x)

        if self.type == "water":
            x = self.projection(self.itransformer(x)[self.pred_len][:,0,:])
            x = self.last(x)

        elif self.type == "rainfall":
            x = self.projection(self.itransformer(x)[self.pred_len])[ :, :, 0]
            x = self.last(x)

        elif self.type == "weather":
            x = self.projection(self.itransformer(x)[self.pred_len])[ :, :, 0]

        return x

water_cfg = {
    "num_variates" : 1,
    "lookback_len" : 8,                 
    "dim" : 64,                          
    "depth" : 6,                          
    "heads" : 4,                                            
    "pred_length" : 1,     
    "num_tokens_per_variate" : 1,     
    "use_reversible_instance_norm" : True 
}

rainfall_cfg = {
    "num_variates" : 48,
    "lookback_len" : 5,                 
    "dim" : 64,                          
    "depth" : 6,                          
    "heads" : 8,                                              
    "pred_length" : 12,     
    "num_tokens_per_variate" : 1,     
    "use_reversible_instance_norm" : True 
}

weather_cfg = {
    "num_variates" : 11,
    "lookback_len" : 31,                 
    "dim" : 64,                          
    "depth" : 6,                          
    "heads" : 8,                                              
    "pred_length" : 3,     
    "num_tokens_per_variate" : 1,     
    "use_reversible_instance_norm" : True 
}

water_itransformer = ITransformer(water_cfg, 1, 64, 1, task="classification", type_="water", proj=True)
rainfall_itransformer = ITransformer(rainfall_cfg, 48, 64, 12, task="regression", type_="rainfall", proj=True)
weather_itransformer = ITransformer(weather_cfg, 11, 64, 3, task="regression", type_="weather", proj=True)

water_itransformer.load_state_dict(torch.load(ITRANSFORMER_WATER))
rainfall_itransformer.load_state_dict(torch.load(ITRANSFORMER_RAINFALL))
weather_itransformer.load_state_dict(torch.load(ITRANSFORMER_WEATHER))

# TIMESNET
class TIMESNET(nn.Module):
    def __init__(self, cfg, task, type_):
        super(TIMESNET, self).__init__()
        self.timesnet = TimesNet(**cfg)
        if task == "classification":
            self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.MSELoss()

        if type_ == "rainfall":
            self.last = nn.Linear(cfg['c_out'], 12)
        elif type_ == "water":
            self.last = nn.Sequential(
                nn.Linear(cfg['c_out'], 1),
                nn.Sigmoid()
            )
        elif type_ == "weather":
            self.last = nn.Linear(cfg['c_out'], 3)

        self.pred_len = cfg["pred_len"]
    
    def forward(self, x):
        x = self.timesnet(x)[:,-self.pred_len:]
        x = self.last(x)[:, :, 0]
        return x

timesnet_rainfall_cfg = {
    "seq_len":5,
    "pred_len":12,
    "c_out":64 ,
    "embed_size":48,
    "k":5,
    "d_model":256,
    "dff":16,
    "num_kernels":6
}

timesnet_water_cfg = {
    "seq_len":8,
    "pred_len":1,
    "c_out":64 ,
    "embed_size":1,
    "k":4,
    "d_model":256,
    "dff":16,
    "num_kernels":6
}

timesnet_weather_cfg = {
    "seq_len":31,
    "pred_len":3,
    "c_out":64 ,
    "embed_size":11,
    "k":5,
    "d_model":256,
    "dff":16,
    "num_kernels":6
}

timesnet_rainfall = TIMESNET(timesnet_rainfall_cfg, "regression", "rainfall")
timesnet_water = TIMESNET(timesnet_water_cfg, "classification", "water")
timesnet_weather = TIMESNET(timesnet_weather_cfg, "regression", "weather")

timesnet_rainfall.load_state_dict(torch.load(TIMESNET_RAINFALL))
timesnet_water.load_state_dict(torch.load(TIMESNET_WATER))
timesnet_weather.load_state_dict(torch.load(TIMESNET_WEATHER))
