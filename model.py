"""
Author: 
Email: 
"""
import torch
from pytorch_wavelets import DWT1DForward
from torch import nn 
import torchvision.transforms as transforms
import torchvision.models as models
from torch.jit import fork, wait
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_size, height, width):
        super(PositionalEncoding2D, self).__init__()
        pe = torch.zeros(embed_size, height, width)
        y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[0::2, :, :] = torch.sin(y_position * div_term[:height].unsqueeze(1))
        pe[1::2, :, :] = torch.cos(x_position * div_term[:width].unsqueeze(0)) 
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe


class FeatureExtractor(nn.Module):
    def __init__(self, cam_num = 1, arm_num = 1):
        super(FeatureExtractor, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2]) 
        for module in self.conv_layers.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        
        self.conv1d_features1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(39)
        )
        self.self_attention1 = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.conv1d_features_oth1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(39)
        )
        if cam_num >=2:
            self.conv1d_features2 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm(39)
            )
            self.self_attention2 = nn.MultiheadAttention(embed_dim=64, num_heads=8) 
            self.conv1d_features_oth2 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm(39)
            )
        if cam_num >=3:
            self.conv1d_features3 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm(39)
            )
            self.self_attention3 = nn.MultiheadAttention(embed_dim=64, num_heads=8)
            self.conv1d_features_oth3 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm(39)
            )
        
        self.position_encoding = PositionalEncoding2D(512, height=15, width=20)
        self.position_encoding1d = PositionalEncoding(d_model = 64)
        
        self.zip1 =  nn.Conv1d(in_channels=300, out_channels=39, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.self_attention_oth1 = nn.MultiheadAttention(embed_dim=64, num_heads=8) 
        
        if arm_num >=2 :
            self.zip2 =  nn.Conv1d(in_channels=300, out_channels=39, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            self.self_attention_oth2 = nn.MultiheadAttention(embed_dim=64, num_heads=8) 
        
        if arm_num >=3 :
            self.zip3 =  nn.Conv1d(in_channels=300, out_channels=39, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            self.self_attention_oth3 = nn.MultiheadAttention(embed_dim=64, num_heads=8) 
        

        
    def forward(self, x):  
        if x.shape[1] == 1:
            x=x.squeeze(dim = 1)
            x, x_oth =self.change2x1(x)

        elif x.shape[1] == 2:
            x1 = x[:, 0, ...]  
            x2 = x[:, 1, ...]
            
            # Using fork parallelization
            task1 = fork(self.change2x1, x1)
            task2 = fork(self.change2x2, x2)

            x11, x12 = wait(task1)
            x21, x22 = wait(task2)
            
            # x = torch.cat([x11, x12], dim= -1)
            # x_oth = torch.cat([x21, x22], dim= -1)

            x = torch.cat([x11, x21], dim= -1)
            x_oth = torch.cat([x12, x22], dim= -1)

        elif x.shape[1] == 3:
            x1 = x[:, 0, ...]  
            x2 = x[:, 1, ...]
            x3 = x[:, 3, ...]
            
            # Using fork parallelization
            task1 = fork(self.change2x1, x1)
            task2 = fork(self.change2x2, x2)
            task3 = fork(self.change2x3, x3)

            x11, x12 = wait(task1)
            x21, x22 = wait(task2)
            x31, x32 = wait(task3)
            
            x = torch.cat([x11, x21, x31], dim= -1)
            x_oth = torch.cat([x12, x22, x32], dim= -1)
            
        return x, x_oth
        
    def change2x1(self, x):
        x = self.conv_layers(x)
        x = self.position_encoding(x)
        x = x.view(x.shape[0], 512, -1) 
        x = x.transpose(1, 2)
        x = self.zip1(x)
        x = x.transpose(1, 2)
        x = x.permute(0, 2, 1)
        x_change = x

        x = self.conv1d_features1(x.permute(0, 2, 1)) 
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2) 
        x = self.position_encoding1d(x)
        x, _ = self.self_attention1(x, x, x)
        x = x.permute(1, 0, 2) 

        '''
        If using a dual arm model, please remove the following comments 
        and modify the return value to "return x, x_oth"
        '''
        # x_oth = self.conv1d_features_oth1(x_change.permute(0, 2, 1))  
        # x_oth = x_oth.permute(0, 2, 1) 
        # x_oth = x_oth.permute(1, 0, 2) 
        # x_oth = self.position_encoding1d(x_oth)
        # x_oth, _ = self.self_attention_oth1(x_oth, x_oth, x_oth)
        # x_oth = x_oth.permute(1, 0, 2) 
        return x, x
    def change2x2(self, x):
        x = self.conv_layers(x)
        x = self.position_encoding(x)
        x = x.view(x.shape[0], 512, -1) 
        x = x.transpose(1, 2)
        x = self.zip2(x)
        x = x.transpose(1, 2)
        x = x.permute(0, 2, 1)
        x_change = x

        x = self.conv1d_features2(x.permute(0, 2, 1)) 
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2) 
        x = self.position_encoding1d(x)
        x, _ = self.self_attention2(x, x, x) 
        x = x.permute(1, 0, 2) 

        '''
        If using a dual arm model, please remove the following comments 
        and modify the return value to "return x, x_oth"
        '''
        # x_oth = self.conv1d_features_oth2(x_change.permute(0, 2, 1))  
        # x_oth = x_oth.permute(0, 2, 1)  
        # x_oth = x_oth.permute(1, 0, 2) 
        # x_oth = self.position_encoding1d(x_oth)
        # x_oth, _ = self.self_attention_oth2(x_oth, x_oth, x_oth)
        # x_oth = x_oth.permute(1, 0, 2) 
        return x, x
    
    def change2x3(self, x):
        x = self.conv_layers(x)
        x = self.position_encoding(x)
        x = x.view(x.shape[0], 512, -1) 
        x = x.transpose(1, 2)
        x = self.zip3(x)
        x = x.transpose(1, 2)
        x = x.permute(0, 2, 1)  
        x_change = x

        x = self.conv1d_features3(x.permute(0, 2, 1)) 
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2) 
        x = self.position_encoding1d(x)
        x, _ = self.self_attention3(x, x, x)
        x = x.permute(1, 0, 2) 

        '''
        If using a dual arm model, please remove the following comments 
        and modify the return value to "return x, x_oth"
        '''
        # x_oth = self.conv1d_features_oth3(x_change.permute(0, 2, 1))  
        # x_oth = x_oth.permute(0, 2, 1) 
        # x_oth = x_oth.permute(1, 0, 2) 
        # x_oth = self.position_encoding1d(x_oth)
        # x_oth, _ = self.self_attention_oth3(x_oth, x_oth, x_oth)
        # x_oth = x_oth.permute(1, 0, 2) 
        return x, x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Wavelet_Policy(nn.Module):
    def __init__(self, n_inp = 14, n_oup = 14, his_step = 0, n_embding=64, en_layers=4, de_layers=1, proj='linear',
                 activation='relu', maxlevel=1, en_dropout=0.,
                 de_dropout=0., out_split=False, decoder_init_zero=False, bias=False, wave='haar', se_skip=True,
                 attn_conv_params=None, cam_num = 1, arm_num = 1):
        """

        """
        super(Wavelet_Policy, self).__init__()
        self.args = locals()
        self.n_embding = n_embding
        self.n_oup = n_oup
        self.maxlevel = maxlevel
        self.decoder_init_zero = decoder_init_zero
        self.de_layers = de_layers
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) 
        self.dwt = DWT1DForward(wave = wave, J=maxlevel, mode='symmetric').cuda()
        self.model = FeatureExtractor(cam_num=cam_num, arm_num=arm_num)
        self.upsample_layer = nn.ConvTranspose2d(n_inp, 1, kernel_size=(n_inp, 1), stride=(n_inp, 1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embding, nhead=16, dropout=en_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=en_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=n_embding, nhead=16, dropout=de_dropout)
        self.decoders = nn.ModuleList([nn.TransformerDecoder(decoder_layer, num_layers=en_layers) for _ in range(1 + maxlevel)])
        self.LNs = nn.ModuleList([nn.LayerNorm(n_embding) for _ in range(1 + maxlevel)])
        self.oup_embdings = nn.ModuleList(
            [Embed(n_embding, n_oup, bias=True, proj=proj) for _ in range(1 + maxlevel)])
        self.positional_encoding = PositionalEncoding(d_model=n_embding, max_len=5000)

        if arm_num ==2:
            encoder_layer_oth = nn.TransformerEncoderLayer(d_model=n_embding, nhead=16, dropout=en_dropout)
            self.encoder_oth = nn.TransformerEncoder(encoder_layer_oth, num_layers=en_layers)
            decoder_layer_oth = nn.TransformerDecoderLayer(d_model=n_embding, nhead=16, dropout=de_dropout)
            self.decoders_oth = nn.ModuleList([nn.TransformerDecoder(decoder_layer_oth, num_layers=en_layers) for _ in range(1 + maxlevel)])
            self.LNs_oth = nn.ModuleList([nn.LayerNorm(n_embding) for _ in range(1 + maxlevel)])
            self.oup_embdings_oth = nn.ModuleList(
                [Embed(n_embding, n_oup, bias=True, proj=proj) for _ in range(1 + maxlevel)])
        
        


    def forward(self, inp, img, action_real = None, img_start = None):
        """

        """
        img = self.normalize(img)
        output_features, x_oth= self.model(img)
        embdings1 = output_features
        embdings = embdings1
        embdings = embdings.transpose(0, 1) 
        embdings = embdings.transpose(0, 1)
        dwtt = self.dwt(embdings.transpose(1, 2))
        ttt = [dwtt[1][0].transpose(1, 2), dwtt[1][1].transpose(1, 2), dwtt[1][2].transpose(1, 2), dwtt[0].transpose(1, 2)]
        embdings = embdings.permute(1, 0, 2)
        embdings = self.positional_encoding(embdings)
        en_oup = self.encoder(embdings)
        memory = en_oup
        en_oup = en_oup.permute(1, 0, 2)
        all_de_oup_set = []
        all_scores_set = []

        for i, (decoder, LN) in enumerate(zip( self.decoders, self.LNs)):
            weight = en_oup[:,0:1,:]
            de_inp = ttt[i]
            de_inp = de_inp.permute(1, 0, 2) 
            de_inp = self.positional_encoding(de_inp)
            de_oup_set = decoder(de_inp, memory) 
            de_oup_set = de_oup_set.permute(1, 0, 2)
            all_de_oup_set.append(LN(de_oup_set))
            all_scores_set.append(weight)
        all_scores_set = torch.cat(all_scores_set, dim=1)
        
        coef_set = [linear(all_de_oup_set[i]) for i, linear in enumerate(self.oup_embdings)]
        coef_set_oth = coef_set

        if self.n_oup == 7 : 
            embdings1 = x_oth
            embdings = embdings1
            embdings = embdings.transpose(0, 1) 
            embdings = embdings.transpose(0, 1)
            dwtt = self.dwt(embdings.transpose(1, 2))
            ttt = [dwtt[1][0].transpose(1, 2), dwtt[1][1].transpose(1, 2), dwtt[1][2].transpose(1, 2), dwtt[0].transpose(1, 2)]
            embdings = embdings.permute(1, 0, 2) 
            embdings = self.positional_encoding(embdings)
            en_oup = self.encoder_oth(embdings)
            memory = en_oup
            en_oup = en_oup.permute(1, 0, 2)
            all_de_oup_set = []
            all_scores_set = []

            for i, (decoder, LN) in enumerate(zip( self.decoders_oth, self.LNs_oth)):
                weight = en_oup[:,0:1,:]
                de_inp = ttt[i]
                de_inp = de_inp.permute(1, 0, 2) 
                de_inp = self.positional_encoding(de_inp)
                de_oup_set = decoder(de_inp, memory) 
                de_oup_set = de_oup_set.permute(1, 0, 2)
                all_de_oup_set.append(LN(de_oup_set))
                all_scores_set.append(weight)

            all_scores_set = torch.cat(all_scores_set, dim=1)

            coef_set_oth = [linear(all_de_oup_set[i]) for i, linear in enumerate(self.oup_embdings_oth)]
            
        return coef_set, coef_set_oth, all_scores_set 


class Embed(nn.Module):
    def __init__(self, in_features, out_features, bias=True, proj='linear'):
        super(Embed, self).__init__()
        self.proj = proj
        if proj == 'linear':
            self.embed = nn.Linear(in_features, out_features, bias)
        else:
            self.embed = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1,
                                   padding=1,
                                   padding_mode='replicate', bias=bias)
    def forward(self, inp):
        if self.proj == 'linear':
            inp = self.embed(inp)
        else:
            inp = self.embed(inp.transpose(1, 2)).transpose(1, 2)
        return inp

class EnhancedBlock(nn.Module): # LFDF
    def __init__(self, hid_size, channel, skip=True):
        super(EnhancedBlock, self).__init__()
        self.skip = skip
        self.comp = nn.Sequential(nn.Linear(hid_size, hid_size // 2, bias=False),
                                  nn.ReLU(),
                                  nn.Linear(hid_size // 2, 1, bias=False))
        self.activate = nn.Sequential(nn.Linear(channel, channel // 2, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(channel // 2, channel, bias=False),
                                      nn.Sigmoid())
    def forward(self, inp):
        S = self.comp(inp) 
        E = self.activate(S.transpose(1, 2)) 
        out = inp * E.transpose(1, 2).expand_as(inp)
        if self.skip:
            out += inp
        return out, E  
