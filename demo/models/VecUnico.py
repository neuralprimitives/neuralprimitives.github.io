import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
from .build import MODELS
from models.Transformer_utils import *
from utils import misc
from models.vec_layers import *
import copy
from scipy.optimize import linear_sum_assignment
from pointnet2_ops import pointnet2_utils
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_, ChamferDistanceL1_instance2


def batch_dice_loss(inputs, targets, unique_gt_indices):
    id2idx = {v.item(): i for i, v in enumerate(unique_gt_indices)}
    remapped = torch.tensor([id2idx[t.item()] for t in targets], device=targets.device)
    targets = F.one_hot(remapped, num_classes=len(unique_gt_indices)).permute(1, 0).float()  # shape: [m, N]
    inputs = inputs.sigmoid() # m, 512
    numerator = 2 * torch.einsum("mc,nc->mn", inputs, targets)  # m, 512
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]  # m, 25
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss 


def batch_sigmoid_ce_loss(inputs, targets, unique_gt_indices):
    point_query_number = inputs.shape[1] 
    id2idx = {v.item(): i for i, v in enumerate(unique_gt_indices)}
    remapped = torch.tensor([id2idx[t.item()] for t in targets], device=targets.device)
    targets = F.one_hot(remapped, num_classes=len(unique_gt_indices)).permute(1, 0).float()  # shape: [m, N]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / point_query_number





class VecSelfAttnBlockApi(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = False, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        init_values=None,
        scale_factor=None,
        drop_path: float = 0.0, 
        act_layer=None,  
        norm_layer=nn.LayerNorm, 
        block_style: str = "vnattn-vngraph",  
        combine_style: str = "concat",
        k: int = 10, 
        n_group: int = 2,
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        super().__init__()

        # 仅支持 concat
        if combine_style != "concat":
            raise ValueError(f"Unsupported combine_style: {combine_style}, only 'concat' is allowed.")

        self.norm1 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        self.norm2 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon) 
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.scale_factor = scale_factor


        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2) 

        self.mlp = nn.Sequential(
            VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon = bias_epsilon),
            VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon = bias_epsilon),
            VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon = bias_epsilon),
        )      

        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}.")

        self.attn = None
        self.local_attn = None

        for block_token in block_tokens:
            if block_token == "vnattn":
                self.attn = VNAttention(dim, num_heads=num_heads, mode= "so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                self.local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon=bias_epsilon)
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        self.block_length = len(block_tokens)

        if self.attn and self.local_attn:
            self.merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        attn_features = []

        norm_x = self.norm1(x) 

        if self.attn:
            attn_features.append(self.attn(norm_x))
        if self.local_attn:
            attn_features.append(self.local_attn(norm_x, pos, idx=idx))

        if len(attn_features) == 1:  
            x = x + self.ls1(transform_restore(x, attn_features[0], self.scale_factor))
        elif len(attn_features) == 2:
            merged_features = torch.cat(attn_features, dim=-2)
            merged_features = self.merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.ls1(transform_restore(x, merged_features, self.scale_factor))
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")

        x = x + self.ls2(transform_restore(x, self.mlp(self.norm2(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.scale_factor))
        return x

class VecCrossAttnBlockApi(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = False, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        init_values=None,
        scale_factor=None,
        drop_path: float = 0.0, 
        act_layer=None, 
        norm_layer=nn.LayerNorm, 
        self_attn_block_style: str = "attn-deform", 
        self_attn_combine_style: str = "concat",
        cross_attn_block_style: str = "attn-deform", 
        cross_attn_combine_style: str = "concat",
        k: int = 10, 
        n_group: int = 2,
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        super().__init__()        

        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2)
        if "vn" in self_attn_block_style or "vn" in cross_attn_block_style:
            self.norm1 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
            self.norm2 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            
        self.mode = mode
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.scale_factor = scale_factor

        if "vn" in self_attn_block_style or "vn" in cross_attn_block_style:
            self.mlp = nn.Sequential(
                VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon=bias_epsilon),
                VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon=bias_epsilon),
                VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon=bias_epsilon),
            )
        else:       
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                act_layer,
                nn.Linear(int(dim * mlp_ratio), dim),
            )

        self.self_attn, self.local_self_attn, self.self_attn_merge_map = self.build_attention(
            block_style=self_attn_block_style, 
            combine_style=self_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            bias_epsilon=bias_epsilon
        )

        self.cross_attn, self.local_cross_attn, self.cross_attn_merge_map = self.build_attention(
            block_style=cross_attn_block_style, 
            combine_style=cross_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            is_cross=True,
            bias_epsilon=bias_epsilon
        )
        if "vn" in self_attn_block_style or "vn" in cross_attn_block_style:
            self.norm_q = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
            self.norm_v = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        else:
            self.norm_q = norm_layer(dim)
            self.norm_v = norm_layer(dim)

    def build_attention(self, block_style, combine_style, dim, num_heads, k, is_cross=False, bias_epsilon=1e-6):
        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}. Expected 'attn', 'deform', or both.")

        attn = None
        local_attn = None

        for block_token in block_tokens:
            if block_token == "vnattn":
                attn = VNAttention(dim, num_heads=num_heads, mode = "so3") if not is_cross else VNCrossAttention(dim, dim, num_heads=num_heads, mode = "so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon= bias_epsilon) 
            elif block_token == "attn":
                attn = Attention(dim, num_heads=num_heads) if not is_cross else CrossAttention(dim, dim, num_heads=num_heads)
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        merge_map = None
        if attn and local_attn and combine_style == "concat" and "vn"  in block_style:
            merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)
        elif attn and local_attn and combine_style == "concat" and "vn" not in block_style:
            merge_map = nn.Linear(dim * 2, dim) 

        return attn, local_attn, merge_map

    def apply_attention(self, x, pos, idx, attn, local_attn, merge_map, layer_scale , scale_factor = None,mask=None, is_cross=False, v=None, v_pos=None, plane_decoder=False):
        attn_features = []
        norm_x = self.norm1(x) if not is_cross else self.norm_q(x)
        norm_v = self.norm_v(v) if is_cross else None

        if attn:
            attn_features.append(attn(norm_x, mask=mask) if not is_cross else attn(norm_x, norm_v))
        if local_attn:
            attn_features.append(
                local_attn(norm_x, pos, idx=idx) if not is_cross else 
                local_attn(q=norm_x, v=norm_v, q_pos=pos, v_pos=v_pos, idx=idx)
            )

        if len(attn_features) == 1 and not plane_decoder:
            return x + layer_scale(transform_restore(x, attn_features[0], scale_factor))
        elif len(attn_features) == 1 and  plane_decoder:
            return x + layer_scale(attn_features[0])   
        elif len(attn_features) == 2 and merge_map and not plane_decoder:
            merged_features = torch.cat(attn_features, dim=-2)
            merged_features = merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x + layer_scale(transform_restore(x, merged_features, scale_factor))
        elif len(attn_features) == 2 and merge_map and plane_decoder:
            merged_features = torch.cat(attn_features, dim=-1)
            merged_features = merge_map(merged_features)
            return x + layer_scale(merged_features)
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None, plane_decoder=False):
        
        if not plane_decoder:
            mask = None
            if denoise_length is not None:
                query_len = q.size(1)
                mask = torch.zeros(query_len, query_len).to(q.device)
                mask[:-denoise_length, -denoise_length:] = 1.

            q = self.apply_attention(
                x=q, pos=q_pos, idx=self_attn_idx, 
                attn=self.self_attn, local_attn=self.local_self_attn, 
                merge_map=self.self_attn_merge_map, mask=mask,
                layer_scale=self.ls1, scale_factor=self.scale_factor
            ) 

            q = self.apply_attention(
                x=q, v=v, pos=q_pos, v_pos=v_pos, idx=cross_attn_idx, 
                attn=self.cross_attn, local_attn=self.local_cross_attn, 
                merge_map=self.cross_attn_merge_map, is_cross=True,
                layer_scale=self.ls2, scale_factor=self.scale_factor
            )

            q = q + self.ls2(transform_restore(q, self.mlp(self.norm2(q).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.scale_factor))
            return q
        else:
            mask = None

            q = self.apply_attention(
                x=q, v=v, pos=q_pos, v_pos=v_pos, idx=cross_attn_idx, 
                attn=self.cross_attn, local_attn=self.local_cross_attn, 
                merge_map=self.cross_attn_merge_map, is_cross=True,
                layer_scale=self.ls2, scale_factor=self.scale_factor, plane_decoder=plane_decoder
            )

            q = self.apply_attention(
                x=q, pos=q_pos, idx=self_attn_idx, 
                attn=self.self_attn, local_attn=self.local_self_attn, 
                merge_map=self.self_attn_merge_map, mask=mask,
                layer_scale=self.ls1, scale_factor=self.scale_factor, plane_decoder=plane_decoder
            ) 
            
            q = q + self.ls2(self.mlp(self.norm2(q)))
            return q
            

######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm,
        block_style_list=['vnattn-vngraph'], combine_style='concat', k=10, n_group=2, mode="so3",bias_epsilon=1e-6,scale_factor=None):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VecSelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group, mode=mode, bias_epsilon=bias_epsilon, scale_factor=scale_factor
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VecCrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group, mode=mode, bias_epsilon=bias_epsilon, scale_factor=scale_factor
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None, plane_decoder=False):
        if not plane_decoder:
            if denoise_length is None:
                self_attn_idx = knn_point(self.k, q_pos, q_pos)
            else:
                self_attn_idx = None
            cross_attn_idx = knn_point(self.k, v_pos, q_pos)
            for _, block in enumerate(self.blocks):
                q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        else:
            self_attn_idx = None
            cross_attn_idx = None
            for _, block in enumerate(self.blocks):
                q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length, plane_decoder=plane_decoder)
        return q

class PointTransformerEncoder(nn.Module):
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            # print("init vec linear")
            trunc_normal_(m.weight, std=.02)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            # print("init vec linear")
            trunc_normal_(m.weight, std=.02)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None, plane_decoder=False):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length, plane_decoder=plane_decoder)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))



class VecDGCNN(nn.Module):
    def __init__(self, k = 16, bias_epsilon=1e-6):

        super().__init__()
        self.k = k
        
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.conv1 = VecLinearNormalizeActivate(2, 32, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv2 = VecLinearNormalizeActivate(64, 64, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv3 = VecLinearNormalizeActivate(128, 64, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv4 = VecLinearNormalizeActivate(128, 128, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)

        self.pool1 = VecMaxPool(32, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool2 = VecMaxPool(64, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool3 = VecMaxPool(64, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool4 = VecMaxPool(128, mode="sim3", bias_epsilon=bias_epsilon)
    
        self.num_features = 128
        
    @staticmethod
    def fps_downsample(coor, x, num_group):
        
        xyz = coor.transpose(1, 2).contiguous()  # [B, N, 3]
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)  # [B, num_group]

        B, C, D, N = x.shape  # D = 3
        x_reshaped = x.view(B, C * D, N)  
        combined_x = torch.cat([coor, x_reshaped], dim=1)  # [B, 3 + C*3, N]
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)  # [B, 3 + C*3, num_group]
        new_coor = new_combined_x[:, :3, :]  # [B, 3, num_group]
        new_x = new_combined_x[:, 3:, :].view(B, C, D, num_group)  # [B, C, 3, num_group]

        return new_coor, new_x


    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        bias = x_k.mean(dim=-1, keepdim=True)
         
        x_q = x_q  - bias
        x_k = x_k  - bias
        k = self.k
        batch_size = x_k.size(0) # bs, 1, 3, N
        num_points_k = x_k.size(-1)
        num_points_q = x_q.size(-1)
        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        
           
        num_dims = x_k.size(1)
        feature = x_k.permute(0, 3, 1, 2).contiguous().view(batch_size * num_points_k, num_dims, -1)[idx, :, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims, -1).permute(0, 3, 4,2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, 3, num_points_q, 1).expand(-1, -1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        
        feature = feature + torch.cat((bias, bias), dim=1).unsqueeze(-1)
        return feature
 

    def forward(self, x, num):
        
        coor = x.transpose(-1, -2).contiguous() #bs,3,N
        x = x.transpose(-1, -2).unsqueeze(1).contiguous() # bs,1,3,N
   
        f = self.get_graph_feature(coor, x, coor, x)
        f = self.conv1(f)
        f = self.pool1(f)
        
        coor_q, f_q = self.fps_downsample(coor, f, num_group=num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        
        f = self.conv2(f)
        f = self.pool2(f)
        coor = coor_q

        f= self.get_graph_feature(coor, f, coor, f)
        f = self.conv3(f)
        f = self.pool3(f)

        coor_q, f_q = self.fps_downsample(coor, f, num_group=num[1])
        f= self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.conv4(f)
        f = self.pool4(f)
        
        coor = coor_q.squeeze(1).transpose(2, 1)
        
        return coor, f

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512, bias_epsilon=1e-6):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        
        self.pool = VecMaxPool(input_dims//2, mode="sim3", bias_epsilon=bias_epsilon)

        self.layer = nn.Sequential(
            VecLinear(v_in=input_dims, v_out=hidden_dim, mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(
                hidden_dim, act_func=nn.LeakyReLU(negative_slope=0.2), mode="so3", bias_epsilon=bias_epsilon
            ),
            VecLinear(v_in=hidden_dim, v_out=step, mode="so3", bias_epsilon=bias_epsilon),
        )

    def forward(self, rec_feature):
        batch_size = rec_feature.size(0)

        g_feature = self.pool(rec_feature.permute(0,2,3,1))
        token_feature = rec_feature

        patch_feature = torch.cat(
            [
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1, -1),
                token_feature,
            ],
            dim=-2,
        )

        patch_feature = patch_feature - patch_feature.mean(dim=1, keepdim=True)
        rebuild_pc = (
            self.layer(patch_feature.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .reshape(batch_size, -1, self.step, 3)
        )
        assert rebuild_pc.size(1) == rec_feature.size(1)

        rebuild_pc = rebuild_pc
        return rebuild_pc

class QueryRanking(nn.Module):
    def __init__(self, mode = "sim3", bias_epsilon=1e-6):
        super().__init__()
        self.fc_inv = VecLinear(1, 256, mode="so3", bias_epsilon=bias_epsilon)
        self.fc_O = VecLinear(1024, 256, mode="sim3", bias_epsilon=bias_epsilon)
        self.fc_bias = VecLinear(1024, 256, mode="sim3", bias_epsilon=bias_epsilon)
        self.query_ranking = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, global_feature, x):

        bias = self.fc_bias(global_feature) 
        z_so3 = self.fc_O(global_feature).unsqueeze(-1) - bias.unsqueeze(-1)
        x = x - x.mean(dim=1, keepdim=True)
        v_inv_per_point = self.fc_inv(x.unsqueeze(-2).permute(0, 2, 3, 1)) 

        
        v_inv_per_point = (v_inv_per_point * z_so3).sum(-2).permute(0, 2, 1) # b, 512, c, 1
        v_inv_per_point = v_inv_per_point / (v_inv_per_point.norm(dim=-1, keepdim=True) + 1e-6)
        x = self.query_ranking(v_inv_per_point)

        idx = torch.argsort(x, dim=1, descending=True) 
        
        return x

class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        self.denoise_length = getattr(config, 'denoise_length', 64)
        self.query_selection = getattr(config, 'query_selection', True)
        self.mode = getattr(config, 'mode', "sim3")
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        
        assert self.encoder_type in ['vecgraph'], f'unexpected encoder_type {self.encoder_type}'
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim


        if self.encoder_type == 'vecgraph':
            self.grouper = VecDGCNN(bias_epsilon=self.bias_epsilon)
        else:
            raise NotImplementedError(f'encoder_type {self.encoder_type} not implemented')
        self.pos_embed = nn.Sequential(
            VecLinear(v_in=1, v_out=128, mode="so3", bias_epsilon=self.bias_epsilon),
            VecActivation(128, act_func=act_func, mode="so3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=128, v_out=encoder_config.embed_dim, mode="so3", bias_epsilon=self.bias_epsilon),
        )  
        
        self.input_proj = nn.Sequential(
            VecLinear(v_in=self.grouper.num_features, v_out=512, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(512, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=512, v_out=encoder_config.embed_dim, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        
        
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        
        self.increase_dim = nn.Sequential(
            VecLinear(v_in=encoder_config.embed_dim, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=global_feature_dim, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        self.pool = VecMaxPool(global_feature_dim, mode="sim3", bias_epsilon=self.bias_epsilon)
        
  
        self.coarse_pred = nn.Sequential(
            VecLinear(v_in=global_feature_dim, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=query_num, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        
        self.mlp_query = nn.Sequential(
            VecLinear(v_in=global_feature_dim + 1, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(
                v_in=1024, v_out=decoder_config.embed_dim, mode="sim3", bias_epsilon=self.bias_epsilon
            )
        )
        
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = VecLinear(
                    v_in=encoder_config.embed_dim,
                    v_out=decoder_config.embed_dim,
                    mode="sim3",
                    bias_epsilon=self.bias_epsilon,
                )
    
    
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = QueryRanking(self.mode, self.bias_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        
        coor, f = self.grouper(xyz, self.center_num) # b n c
        
        x = self.input_proj(f) 
        
        coor_mean = coor.mean(dim=1, keepdim=True)
        pe = self.pos_embed(coor.permute(0, 2, 1).unsqueeze(1).contiguous()-coor_mean.permute(0, 2, 1).unsqueeze(1).contiguous())
        x = (x + pe).permute(0, 3, 1, 2).contiguous() 
        x = self.encoder(x, coor)
        
        
        global_feature = self.increase_dim(x.permute(0, 2, 3, 1).contiguous())
        global_feature = self.pool(global_feature)
        coarse = self.coarse_pred(global_feature)
        
        
        if self.query_selection:
            coarse_inp = misc.fps(xyz.contiguous(), self.num_query // 2)  # B 128 3
            coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 224+128 3
            query_ranking = self.query_ranking(global_feature, coarse)  # b n 1
            idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
            coarse = torch.gather(
                coarse, 1, idx[:, : self.num_query].expand(-1, -1, coarse.size(-1))
            )
            
        mem = self.mem_link(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
     
        if self.training:
            if self.denoise_length > 0:
                picked_points = misc.fps(xyz.contiguous(), self.denoise_length)
                picked_points = misc.jitter_points(picked_points)
                coarse = torch.cat([coarse, picked_points], dim=1)  # B 256+64 3?
                denoise_length = self.denoise_length
            else:
                denoise_length = None

            q = self.mlp_query(
                torch.cat(
                    [
                        global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1, -1),
                        coarse.unsqueeze(2),
                    ],
                    dim=-2,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )  # b n c
            q = q.permute(0, 3, 1, 2).contiguous()
            
            q = self.decoder(
                q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length
            )
            
            return q, coarse, self.denoise_length
        else:
            # produce query
            q = self.mlp_query(
                torch.cat(
                    [
                        global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1, -1),
                        coarse.unsqueeze(2),
                    ],
                    dim=-2,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )  # b n c
            q = q.permute(0, 3, 1, 2).contiguous()
            

            q = self.decoder(
                q=q, v=mem, q_pos=coarse, v_pos=coor
            )
            return q, coarse, 0


class Primitive_Segmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.primitive_query_type = config.primitive_query_type
        assert self.primitive_query_type in ['static'], f'unexpected plane_query_type {self.plane_query_type}, static query cannot gurantee sim3 invariance'
        self.primitive_query_num = config.primitive_query_num
        primitive_decoder_cofig = config.primitive_decoder_cofig
        hidden_dim = primitive_decoder_cofig.embed_dim
        self.num_decoders = primitive_decoder_cofig.depth
        self.first_stage, self.second_stage = getattr(config, 'first_stage', 199), getattr(config, 'second_stage', 300)
        trans_dim = config.trans_dim
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        self.num_equation_params = getattr(config, 'num_equation_params', 3)
        self.num_primitive_classes = getattr(config, 'num_primitive_classes', 5)
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        self.query_feat = nn.Embedding(self.primitive_query_num, primitive_decoder_cofig.embed_dim)
           

        self.point_feature_norm = VecLayerNorm(trans_dim, mode="sim3", eps=self.bias_epsilon)     
        self.std_point = nn.Sequential(
            VecLinear(v_in=trans_dim, v_out=trans_dim//2, mode="so3", bias_epsilon=self.bias_epsilon),
            VecActivation(trans_dim//2, act_func=act_func, mode="so3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=trans_dim//2, v_out=trans_dim//4, mode="so3", bias_epsilon=self.bias_epsilon),
            VecActivation(trans_dim//4, act_func=act_func, mode="so3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=trans_dim//4, v_out=3, mode="so3", bias_epsilon=self.bias_epsilon),
        )    
        
        self.simple_rot = VecLinear(v_in=trans_dim, v_out=3, mode="so3", bias_epsilon=self.bias_epsilon)
        self.point_feature_encoder = nn.Linear(
                in_features=trans_dim*3,
                out_features=hidden_dim)
        
        self.class_embed_head = nn.Linear(hidden_dim, self.num_primitive_classes)
        self.mask_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim)) 
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.quadric_head = nn.Linear(hidden_dim, self.num_equation_params)

        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_decoders):
            layer_config = copy.deepcopy(primitive_decoder_cofig)
            layer_config.depth = 1
            layer_config.self_attn_block_style_list = [primitive_decoder_cofig.self_attn_block_style_list[i]]
            layer_config.cross_attn_block_style_list = [primitive_decoder_cofig.cross_attn_block_style_list[i]]
            self.decoder_layers.append(PointTransformerDecoderEntry(layer_config))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_module(self, query_feat, mask_features):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        output_class = self.class_embed_head(query_feat)
        output_masks = torch.einsum('bic,bjc->bij', mask_embed, mask_features)
        
        return output_class, output_masks

    
    @staticmethod
    def _canonicalize_params(params: torch.Tensor) -> torch.Tensor:
        norm = params.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return params / norm

    def get_primitive_query(self, ret):
        pred_masks = ret['pred_masks'].sigmoid()  
        rebuild_points = ret['rebuild_points']    
        factor = rebuild_points.shape[1] // pred_masks.shape[2]  # 32
        pred_masks = pred_masks.unsqueeze(-1).expand(-1, -1, -1, factor).reshape(
            pred_masks.shape[0], pred_masks.shape[1], -1)  # b, num_queries, 16384
        return ret
    
    def forward(self, ret, epoch=1):
        coarse_point_cloud, rebuild_points, point_features = ret["coarse_point_cloud"], ret["rebuild_points"], ret["point_features"]
        batch_size = coarse_point_cloud.shape[0]
        queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        point_features = self.point_feature_norm(point_features) # b,n,c,3
        z0 = self.std_point(point_features.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        
        point_features_mean = point_features.mean(dim=1, keepdim=False)  # b,1,c,3
        z_rotation = self.simple_rot(point_features_mean)
    
        point_features = torch.einsum('bijm,bikm->bijk', point_features, z0)
            
        point_features = self.point_feature_encoder(point_features.view(batch_size, point_features.size(1), -1))
        for decoder_counter in range(self.num_decoders):
            queries = self.decoder_layers[decoder_counter](queries, point_features, q_pos=None, v_pos=None, plane_decoder=True)
         
        output_class, output_masks = self.mask_module(queries, point_features)
        

        predicted_params = self.quadric_head(queries)
        predicted_params = torch.einsum('bij,bni->bnj', z_rotation, predicted_params)
        predicted_params = self._canonicalize_params(predicted_params)  # B, num_queries, 10

        if self.training:
            ret = {
                "coarse_point_cloud": coarse_point_cloud,  
                "rebuild_points": rebuild_points,         
                "class_prob": output_class,                
                "pred_masks": output_masks,                
                "denoised_coarse": ret["denoised_coarse"], 
                "denoised_fine": ret["denoised_fine"],    
                "quadrics": predicted_params             
            }
        else:
            ret = {
                "coarse_point_cloud": coarse_point_cloud,
                "rebuild_points": rebuild_points,
                "class_prob": output_class,
                "pred_masks": output_masks,
                "quadrics": predicted_params
            }
        
        if epoch >= self.first_stage:
            ret = self.get_primitive_query(ret)
         
        return ret



@MODELS.register_module()
class VecUnico(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        self.plane_query_num = getattr(config, 'plane_query_num', 40)

        
        self.first_stage = getattr(config, 'first_stage', 200)


        self.fold_step = 8
        self.base_model = PCTransformer(config)
        
        
        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query, bias_epsilon=self.bias_epsilon)
        else:
            self.factor = self.fold_step**2
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2, bias_epsilon=self.bias_epsilon)
        
        self.increase_dim = nn.Sequential(
            VecLinear(self.trans_dim, 1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=nn.LeakyReLU(negative_slope=0.2), mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(1024, 1024, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        self.pool = VecMaxPool(1024, mode="sim3", bias_epsilon=self.bias_epsilon)
        self.reduce_map = nn.Sequential(
            VecLinear(self.trans_dim + 1025, self.trans_dim, mode="sim3", bias_epsilon=self.bias_epsilon)) 
        self.primitive_segmentation = Primitive_Segmentation(config.primitive_segmentation_config)


    def get_segmentation_labels(self, ret, gt, gt_index, coarse=False):
        """
        Generate segmentation labels based on predicted masks.
        coarse: If True, use coarse point cloud; otherwise, use rebuilt points. 
                try not to use coarse point cloud, it is not accurate enough.

        Args:
            ret: Dictionary containing 'pred_masks' with shape (B, N1, N2)

        Returns:
            seg_labels: Tensor of shape (B, N1) with segmentation labels
        """
        
        from extensions.chamfer_dist import ChamferDistanceL1_
        get_index = ChamferDistanceL1_()
            
        if coarse:
            coarse_point_cloud = ret['coarse_point_cloud']  # B, N1, 3
            idx = get_index(coarse_point_cloud, gt) # B, N1
            B, N = idx.shape  # B=5, N=512
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, N)  # shape [5, 512]
            selected_gt = gt_index[batch_indices, idx]  # shape [5, 512]
            ret["instance_lables"] = selected_gt
        else:
            points = ret['rebuild_points']  # B, N1, 3
            idx = get_index(points, gt)  # B, N1
            B, N = idx.shape  # B=5, N=16384
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, N) 
            selected_gt = gt_index[batch_indices, idx]  # shape [5, 16384]
            selected_gt = selected_gt.reshape(B, -1, self.factor)
            patch_index = torch.mode(selected_gt, dim=-1).values
            ret["instance_lables"] = patch_index

        return ret
    
    def _equation_param_cost_matrix(
        self,
        theta_pred: torch.Tensor,   # [Np,10] -> [A,B,C,D,E,F,G,H,I,J]
        theta_gt:   torch.Tensor,   # [Ng,10]
        eps: float = 1e-9,
        beta: float = 1e-2,         # Smooth-L1 (Huber) delta
    ):
        """
        Uniform, scale-invariant, sign-sensitive quadric supervision.

        Canonicalization (both pred & GT):  θ̂ = θ / (||θ||₂ + eps)   (no sign flip)
        Cost: elementwise Smooth-L1(θ̂_pred - θ̂_gt), averaged over the 10 terms.
        Returns: [Np, Ng]
        """
        import torch
        import torch.nn.functional as F

        # ---- sanity on shapes ----
        assert theta_pred.dim() == 2 and theta_pred.size(1) == 3, "theta_pred must be [Np,3]"
        assert theta_gt.dim()   == 2 and theta_gt.size(1)   == 3, "theta_gt must be [Ng,3]"

        # ---- 10-D canonicalization (scale invariance, sign supervised) ----
        s_p = torch.linalg.vector_norm(theta_pred, dim=1).clamp_min(eps)   # [Np]
        s_g = torch.linalg.vector_norm(theta_gt,   dim=1).clamp_min(eps)   # [Ng]
        vec_p = theta_pred / s_p[:, None]   # [Np,10]
        vec_g = theta_gt   / s_g[:, None]   # [Ng,10]

        # ---- pairwise Smooth-L1 on the 4-D vectors ----
        # (θ̂_p - θ̂_g) -> Huber -> mean over terms
        d = vec_p[:, None, :] - vec_g[None, :, :]                 # [Np,Ng,3]
        costs = F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta, reduction='none').mean(dim=2)  # [Np,Ng]

        if self.training:
            if not torch.isfinite(costs).all():
                bad = torch.nonzero(~torch.isfinite(costs), as_tuple=False)[:5]
                raise ValueError(f"Non-finite costs detected (first 5): {bad.tolist()}")

        return costs

    
    def get_loss(self, config, ret, gt , gt_index, gt_coeff, gt_type, epoch = 1, scale = None):
        """
        Compute losses for the model

        Args:
            config: Configuration with loss weights
            ret: prediction
            gt: GT points
            gt_index: GT indices
            gt_coeff: GT coefficients
            gt_type: GT types

        Returns:
            Dictionary of computed losses
        """
        from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_instance2
        loss_func = ChamferDistanceL1()
        
        chamfer_plane = ChamferDistanceL1_instance2()

        # Deterministic: dataset supplies gt_type with a trailing singleton dim -> remove it once.
        gt_type = gt_type.squeeze(-1)
        reconstructed_points, coarse_point_cloud = ret['rebuild_points'],  ret['coarse_point_cloud']
        if scale is not None:
            reconstructed_points = reconstructed_points / scale.unsqueeze(1).unsqueeze(2)
            coarse_point_cloud = coarse_point_cloud / scale.unsqueeze(1).unsqueeze(2)
            gt = gt / scale.unsqueeze(1).unsqueeze(2)
        
        device = reconstructed_points.device
        losses = {}
        if epoch < config['first_stage'] or epoch >= config['second_stage']:
            if self.training:
                denoised_coarse, denoised_fine = ret['denoised_coarse'], ret['denoised_fine']
                idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
                denoised_target = index_points(gt, idx) # B n k 3 
                denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
                assert denoised_target.size(1) == denoised_fine.size(1)
                loss_denoised = loss_func(denoised_fine, denoised_target)
                loss_denoised = loss_denoised * 0.5
                losses["loss_denoised"] = loss_denoised

            loss_coarse = loss_func(coarse_point_cloud, gt)
            loss_fine = loss_func(reconstructed_points, gt)
            loss_recon = loss_coarse + loss_fine
            losses["loss_coarse"] = loss_coarse
            losses["chamfer_norm1_loss"] = loss_fine
            if epoch < config['first_stage'] and self.training:
                losses["total_loss_stage1"] = loss_recon + loss_denoised
        elif epoch >= config['first_stage'] and epoch < config['second_stage']:
            ret = self.get_segmentation_labels(ret, gt, gt_index, coarse=False) 
            losses["classification_loss"] = losses.get("classification_loss", 0.0)
            losses["mask_loss"] = losses.get("mask_loss", 0.0)
            losses["dice_loss"] = losses.get("dice_loss", 0.0)
            
            class_prob, reconstructed_points, pred_masks  = ret['class_prob'], ret['rebuild_points'], ret['pred_masks']
            tgt_ids = ret["instance_lables"]
            batch_size = class_prob.size(0)
            size_total = 0
            for batch_idx in range(batch_size):
                # continue
                unique_gt_indices = torch.unique(gt_index[batch_idx].int())
                unique_gt_indices = unique_gt_indices[unique_gt_indices != -1]
                num_ground_truth_planes = unique_gt_indices.size(0)

                ground_truth_pointclouds = [gt[batch_idx, (gt_index[batch_idx] == idx)].reshape(-1, 3) for idx in unique_gt_indices]  
                reconstructed_pointclouds = reconstructed_points[batch_idx]
                

                _gt_types = torch.zeros_like(unique_gt_indices).long()  # 0 plane, 1 cylinder, 2 sphere, 3 cone
                classification_scores = class_prob[batch_idx]  # [Q,5]
                tgt_id = tgt_ids[batch_idx]

                mask_loss = batch_sigmoid_ce_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)
                dice_loss = batch_dice_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)

                log_probs = classification_scores.log_softmax(dim=-1)  # [Q,5]
                class_cost = -log_probs[:, _gt_types]

                cost_matrix = (class_cost * config.obj_class_loss_weight +
                               mask_loss * config.mask_loss_weight +
                               dice_loss * config.mask_loss_weight)

                # Debug: detect non-finite entries before Hungarian (Stage 3)
                if not torch.isfinite(cost_matrix).all():
                    bad = ~torch.isfinite(cost_matrix)
                    print("[Stage3 Debug] Non-finite cost_matrix entries detected:")
                    print("  batch=", batch_idx,
                          " num_gt=", num_ground_truth_planes,
                          " class_cost_any_nan=", (not torch.isfinite(class_cost).all()),
                          " mask_loss_any_nan=", (not torch.isfinite(mask_loss).all()),
                          " dice_loss_any_nan=", (not torch.isfinite(dice_loss).all()))
                    bad_idx = bad.nonzero()
                    for i in range(min(5, bad_idx.size(0))):
                        pi, gi = bad_idx[i].tolist()
                        print(f"    -> cost_matrix[{pi},{gi}] = {cost_matrix[pi,gi].item()}")
                    cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

                hungarian_assignment = linear_sum_assignment(cost_matrix.detach().cpu().numpy()) if num_ground_truth_planes > 0 else ([], [])
                if num_ground_truth_planes > 0:
                    hungarian_assignment = [torch.tensor(a, dtype=torch.long, device=device) for a in hungarian_assignment]
                    pred_matched_idx, gt_matched_idx = hungarian_assignment
                    mached_mask_loss = mask_loss[pred_matched_idx, gt_matched_idx]
                    matched_dice_loss = dice_loss[pred_matched_idx, gt_matched_idx]
                else:
                    pred_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    gt_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    mached_mask_loss = torch.zeros(0, device=device)
                    matched_dice_loss = torch.zeros(0, device=device)

                targets = torch.full((self.plane_query_num,), 4, dtype=torch.long, device=device)
                if pred_matched_idx.numel() > 0:
                    targets[pred_matched_idx] = _gt_types[gt_matched_idx]
                ce_per_query = F.cross_entropy(classification_scores, targets, reduction='none')
                matched_mask_bool = targets != 4
                matched_class_loss = ce_per_query[matched_mask_bool]
                unmatched_class_loss = ce_per_query[~matched_mask_bool] * config.non_obj_class_loss_weight
                total_classification_loss = torch.cat([matched_class_loss, unmatched_class_loss])

                losses["classification_loss"] += total_classification_loss.sum()
                losses["mask_loss"] += mached_mask_loss.sum()
                losses["dice_loss"] += matched_dice_loss.sum()
                size_total += num_ground_truth_planes
            
            losses["classification_loss"] /= (batch_size * self.plane_query_num)
            losses["mask_loss"] /= size_total
            losses["dice_loss"] /= size_total
            
            if self.training:
                losses["total_loss_stage2"] = config.obj_class_loss_weight * losses["classification_loss"]  + config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])
            else:
                losses["total_loss_stage2"] = config.obj_class_loss_weight * losses["classification_loss"]  + config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])

        
        if epoch >= config['second_stage'] and epoch < config['third_stage']:
             
            ret = self.get_segmentation_labels(ret, gt, gt_index, coarse=False)  # patch-level labels only
            losses["plane_chamfer_loss"] = losses.get("plane_chamfer_loss", 0.0)
            losses["classification_loss"] = losses.get("classification_loss", 0.0)
            losses["plane_normal_loss"] = losses.get("plane_normal_loss", 0.0)
            losses["mask_loss"] = losses.get("mask_loss", 0.0)
            losses["dice_loss"] = losses.get("dice_loss", 0.0)
             
            class_prob, reconstructed_points, pred_masks,predicted_params  = ret['class_prob'], ret['rebuild_points'], ret['pred_masks'], ret['quadrics']
            tgt_ids = ret["instance_lables"]
            batch_size = class_prob.size(0)
            size_total = 0
            for batch_idx in range(batch_size):
                # continue
                unique_gt_indices = torch.unique(gt_index[batch_idx].int())
                unique_gt_indices = unique_gt_indices[unique_gt_indices != -1]
                num_ground_truth_planes = unique_gt_indices.size(0)

                ground_truth_pointclouds = [gt[batch_idx, (gt_index[batch_idx] == idx)].reshape(-1, 3) for idx in unique_gt_indices]  
                reconstructed_pointclouds = reconstructed_points[batch_idx]
                
                # loss term1: Compute Plane Chamfer Distance soft labeled
                mask_weights = pred_masks[batch_idx].sigmoid()  # num_queries, 512
                 
                 
                mask_weights = mask_weights.view(self.plane_query_num, -1, 1).expand(-1, -1, self.factor).reshape(self.plane_query_num, -1)  # num_queries, 512 * factor
                plane_chamfer_distance = chamfer_plane(reconstructed_pointclouds, ground_truth_pointclouds, mask_weights)

                # loss term2: Compute Plane Normal Loss
                _pred_coeff = predicted_params[batch_idx][..., :3].float()  # [num_queries, 3]
                _gt_coeff = gt_coeff[batch_idx, unique_gt_indices][..., :3].float()  # [num_gt, 3]
                plane_normal_loss = self._equation_param_cost_matrix(_pred_coeff, _gt_coeff)


                _gt_types = torch.zeros_like(unique_gt_indices).long()  # 0 plane, 1 cylinder, 2 sphere, 3 cone
                classification_scores = class_prob[batch_idx]  # [Q,5]
                tgt_id = tgt_ids[batch_idx]

                mask_loss = batch_sigmoid_ce_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)
                dice_loss = batch_dice_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)

                log_probs = classification_scores.log_softmax(dim=-1)  # [Q,5]
                class_cost = -log_probs[:, _gt_types]

                cost_matrix = (class_cost * config.obj_class_loss_weight +
                               plane_chamfer_distance * config.plane_chamfer_loss_weight +
                               plane_normal_loss * config.plane_normal_loss_weight +
                               mask_loss * config.mask_loss_weight +
                               dice_loss * config.mask_loss_weight)

                # Debug: detect non-finite entries before Hungarian (Stage 3)
                if not torch.isfinite(cost_matrix).all():
                    bad = ~torch.isfinite(cost_matrix)
                    print("[Stage3 Debug] Non-finite cost_matrix entries detected:")
                    print("  batch=", batch_idx,
                          " num_gt=", num_ground_truth_planes,
                          " class_cost_any_nan=", (not torch.isfinite(class_cost).all()),
                          " chamfer_any_nan=", (not torch.isfinite(plane_chamfer_distance).all()),
                          " quadric_any_nan=", (not torch.isfinite(plane_normal_loss).all()),
                          " mask_loss_any_nan=", (not torch.isfinite(mask_loss).all()),
                          " dice_loss_any_nan=", (not torch.isfinite(dice_loss).all()))
                    bad_idx = bad.nonzero()
                    for i in range(min(5, bad_idx.size(0))):
                        pi, gi = bad_idx[i].tolist()
                        print(f"    -> cost_matrix[{pi},{gi}] = {cost_matrix[pi,gi].item()}")
                    cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

                hungarian_assignment = linear_sum_assignment(cost_matrix.detach().cpu().numpy()) if num_ground_truth_planes > 0 else ([], [])
                if num_ground_truth_planes > 0:
                    hungarian_assignment = [torch.tensor(a, dtype=torch.long, device=device) for a in hungarian_assignment]
                    pred_matched_idx, gt_matched_idx = hungarian_assignment
                    matched_plane_chamfer_distance = plane_chamfer_distance[pred_matched_idx, gt_matched_idx]
                    # if plane_normal_loss_fitted_full is not None:
                    matched_plane_normal_loss = plane_normal_loss[pred_matched_idx, gt_matched_idx]
                    mached_mask_loss = mask_loss[pred_matched_idx, gt_matched_idx]
                    matched_dice_loss = dice_loss[pred_matched_idx, gt_matched_idx]
                else:
                    pred_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    gt_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    matched_plane_chamfer_distance = torch.zeros(0, device=device)
                    matched_plane_normal_loss = torch.zeros(0, device=device)
                    mached_mask_loss = torch.zeros(0, device=device)
                    matched_dice_loss = torch.zeros(0, device=device)

                targets = torch.full((self.plane_query_num,), 4, dtype=torch.long, device=device)
                if pred_matched_idx.numel() > 0:
                    targets[pred_matched_idx] = _gt_types[gt_matched_idx]
                ce_per_query = F.cross_entropy(classification_scores, targets, reduction='none')
                matched_mask_bool = targets != 4
                matched_class_loss = ce_per_query[matched_mask_bool]
                unmatched_class_loss = ce_per_query[~matched_mask_bool] * config.non_obj_class_loss_weight
                total_classification_loss = torch.cat([matched_class_loss, unmatched_class_loss])

                losses["plane_chamfer_loss"] += matched_plane_chamfer_distance.sum()
                losses["classification_loss"] += total_classification_loss.sum()
                losses["mask_loss"] += mached_mask_loss.sum()
                losses["plane_normal_loss"] += matched_plane_normal_loss.sum()
                losses["dice_loss"] += matched_dice_loss.sum()
                size_total += num_ground_truth_planes
            
            losses["classification_loss"] /= (batch_size * self.plane_query_num)
            losses["plane_chamfer_loss"] /= size_total
            losses["mask_loss"] /= size_total
            losses["plane_normal_loss"] /= size_total
            losses["dice_loss"] /= size_total
            
            if self.training:
                losses["total_loss_stage3"] = config.obj_class_loss_weight * losses["classification_loss"] + config.plane_normal_loss_weight * losses["plane_normal_loss"] + config.plane_chamfer_loss_weight * losses["plane_chamfer_loss"] + config.chamfer_norm1_loss_weight * (losses["chamfer_norm1_loss"] + losses["loss_coarse"] + losses["loss_denoised"])+ config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])
            else:
                losses["total_loss_stage3"] = config.obj_class_loss_weight * losses["classification_loss"] + config.plane_normal_loss_weight * losses["plane_normal_loss"] + config.plane_chamfer_loss_weight * losses["plane_chamfer_loss"]  + config.chamfer_norm1_loss_weight * (losses["chamfer_norm1_loss"] + losses["loss_coarse"]) + config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])
        return losses

    def forward(self, xyz, epoch=None):
        
        q, coarse_point_cloud, denoise_length = self.base_model(xyz) # B M C and B M 3
            
        B, M, C, _ = q.shape
        
        global_feature = self.increase_dim(q.permute(0, 2, 3, 1).contiguous())
        global_feature = self.pool(global_feature)
        
        rebuild_feature = torch.cat(
            [
                global_feature.unsqueeze(-3).expand(-1, M, -1, -1),
                q,
                coarse_point_cloud.unsqueeze(-2),
            ],
            dim=-2,
        )  # B M 1027 + C
        
        
        rebuild_feature = self.reduce_map(rebuild_feature.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        relative_xyz = self.decode_head(rebuild_feature)
        # return coarse_point_cloud, relative_xyz
        rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()
            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret =  {"coarse_point_cloud": pred_coarse,
                    "rebuild_points": pred_fine,
                    "denoised_coarse": denoised_coarse,
                    "denoised_fine": denoised_fine}
        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query
            
            ret =  {"coarse_point_cloud": coarse_point_cloud,
                    "rebuild_points": rebuild_points}
        
        if epoch is None or epoch >= self.first_stage:
            ret["point_features"] = q[:, :-denoise_length, :] if self.training else q
            ret = self.primitive_segmentation(ret, epoch=epoch)
        
        return ret
        
 