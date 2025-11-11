import torch.nn as nn
import fairseq
from dataclasses import dataclass, field
import torch.nn.functional as F
import torch
import math

from .mamba_blocks import MixerModel

@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory = dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    
class SSLModel(nn.Module): #W2V
    def __init__(self,device, wav2vec_path):
        super(SSLModel, self).__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        """
        input: (batch_size, seq_len)
        """
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype = input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        extract = self.model(input_tmp, mask = False, features_only = True)
        emb = extract['x']
        layerresult = extract['layer_results']
        return emb, layerresult


class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        bottleneck = channels // SE_ratio
        if bottleneck < 2:
            bottleneck = 2
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x
    

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        weighted_sum = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size,1), dilation=(dilation, 1), padding=(num_pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            initial_value = torch.ones(1, 1, 1, i+2) * (1 / (i+2))
            weighted_sum.append(nn.Parameter(initial_value, requires_grad=True))
        self.weighted_sum = nn.ParameterList(weighted_sum)
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.SELU   = nn.SELU()
        self.width  = width
        self.se     = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.SELU(out)
        out = self.bn1(out).unsqueeze(-1) #bz c T 1

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        for i in range(self.nums):
          sp = torch.cat((sp, spx[i]), -1)

          sp = self.bns[i](self.SELU(self.convs[i](sp)))
          sp_s = sp * self.weighted_sum[i]
          sp_s = torch.sum(sp_s, dim=-1, keepdim=False)

          if i==0:
            out = sp_s
          else:
            out = torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)),1)
        out = self.conv3(out)
        out = self.SELU(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
    

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device, args.wav2vec_path)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + mamba')
        self.first_bn = nn.BatchNorm2d(num_features = 1)
        self.selu = nn.SELU(inplace = True)
        self.config = MambaConfig(d_model = args.emb_size, n_layer = args.num_encoders // 2)
        print(self.config)
        self.conformer = MixerModel(d_model = self.config.d_model, 
                                  n_layer = self.config.n_layer, 
                                  ssm_cfg = self.config.ssm_cfg, 
                                  rms_norm = self.config.rms_norm, 
                                  residual_in_fp32 = self.config.residual_in_fp32, 
                                  fused_add_norm = self.config.fused_add_norm
                                  )
        # new framework
        self.layer_n = 24
        self.Build_in_Res2Nets = nn.ModuleList()
        self.bns = nn.ModuleList()
        C = 1024
        Nes_ratio = [8, 8]
        dilation = 2 # 空洞卷积的扩张率,在不增加参数的情况下扩大感受野,dilation=1: 普通卷积; dilation=2: 卷积核元素间间隔1个位置. 感受野 = (kernel_size - 1) * dilation + 1
        SE_ratio = 8 # SE (Squeeze-and-Excitation) 模块的压缩比,控制通道注意力机制的中间层维度,SE模块中间层维度 = planes / SE_ratio,常用值: 8, 16 等
        for i in range(self.layer_n - 1):
            self.Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio))
            self.bns.append(nn.BatchNorm1d(C))
        self.bn = nn.BatchNorm1d(1024)
        self.SELU = nn.SELU()
        self.new_config = MambaConfig(d_model = C, n_layer = args.num_encoders // 2)
        print(self.new_config)
        self.new_conformer = MixerModel(d_model = self.new_config.d_model, 
                                  n_layer = self.new_config.n_layer, 
                                  ssm_cfg = self.new_config.ssm_cfg, 
                                  rms_norm = self.new_config.rms_norm, 
                                  residual_in_fp32 = self.new_config.residual_in_fp32, 
                                  fused_add_norm = self.new_config.fused_add_norm
                                  )
        # self.fc = nn.Linear(1024, 1)

    def forward_init(self, x):
        """
         Input:  (batch_size, seq_len) [8, 66800]
         
        """
        #-------pre-trained Wav2vec model fine tunning ------------------------#
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        # x_ssl_feat:[batch, 208, 1024] 
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        # x:[batch, 208, 144] 
        x = x.unsqueeze(dim = 1) # add channel #(bs, 1, frame_number, 256)
        # x:[batch, 1, 208, 144]
        x = self.first_bn(x)
        # x:[batch, 1, 208, 144]
        #-------Mamba Encoder ------------------------#
        x = self.selu(x)
        # x:[batch, 1, 208, 144]
        x = x.squeeze(dim = 1)
        # x:[2, 208, 144]
        out = self.conformer(x) 
        # out:[2, 2]
        # tensor([[-0.1186, -0.1083],                                                                                                                                                                       | 0/12690 [00:00<?, ?it/s]tensor([[-0.1186, -0.1083],
        #         [ 0.0400, -0.0670]], device='cuda:0', grad_fn=<AddmmBackward0>)
        return out
    
    def process_layer_results(self, init_layer_results):
        layer_results = []
        for layer in init_layer_results: # layer = (x, z)
            y = layer[0].transpose(0, 1).transpose(1,2) # (T, bz, dim) -> (bz, T, dim) -> (bz, dim, T)
            layer_results.append(y)
        for i in range(self.layer_n - 1):
            if i == 0:
                sp = layer_results[i]
            else:
                sp = sp + layer_results[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.SELU(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                # out = torch.cat((out, sp), 1)
                out = out + sp
        out = out + layer_results[-1] # (bz, dim, T)
        out = self.bn(out)
        out = self.SELU(out)
        out = out.transpose(1, 2) # (bz, dim, T) -> (bz, T, dim)
        out = self.new_conformer(out)
        return out

    def forward(self, x): #(batch_size, seq_len) [8, 66800]
        #-------pre-trained Wav2vec model fine tunning ------------------------#
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        output = self.process_layer_results(layerResult)
        output = output.squeeze() # (bz, 1) -> (bz, )

        return output


