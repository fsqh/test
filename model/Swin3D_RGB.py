import os
import torch
import torch.nn as nn
from Swin3D.models import Swin3DUNet
from MinkowskiEngine import SparseTensor


class Swin3D(nn.Module):
    def __init__(
        self,
        depths,              # 网络各层的深度参数列表
        channels,            # 网络各层的通道数参数列表
        num_heads,           # 注意力机制中的头数
        window_sizes,        # 窗口大小参数列表
        up_k,                # 上采样时的扩展系数
        quant_sizes,         # 量化大小参数列表
        drop_path_rate=0.2,  # 随机深度丢弃率，默认0.2
        num_layers=4,        # 网络层数，默认4层
        num_classes=13,      # 分类类别数，默认13类
        stem_transformer=False,  # 是否在stem层使用transformer，默认False
        upsample="deconv",   # 上采样方法，默认反卷积
        down_stride=2,       # 下采样步长，默认2
        knn_down=True,       # 是否使用KNN下采样，默认True
        signal=True,         # 是否使用信号处理，默认True
        in_channels=6,      # 输入通道数，默认6
        use_offset=False,    # 是否使用偏移量，默认False
        fp16_mode=2,         # 混合精度训练模式，默认2
    ):
        super().__init__()    # 调用父类初始化方法
        self.signal = signal  # 保存信号处理标志
        self.use_offset = use_offset  # 保存偏移量使用标志
        # 初始化Swin3DUNet骨干网络
        self.backbone = Swin3DUNet(
            depths,
            channels,
            num_heads,
            window_sizes,
            quant_sizes,
            up_k=up_k,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
            num_layers=num_layers,
            stem_transformer=stem_transformer,
            upsample=upsample,
            first_down_stride=down_stride,
            knn_down=knn_down,
            in_channels=in_channels,
            cRSE="XYZ_RGB",     # 3D坐标和RGB信息的组合
            fp16_mode=fp16_mode,
        )

    def forward(self, feats, xyz, batch):
        """
        前向预处理：把稠密坐标/特征转换为 MinkowskiEngine 所需的稀疏张量格式，
        并构造额外的几何-颜色张量 coords_sp，供下游 Swin3DUNet 使用。

        参数
        ----
        feats : torch.Tensor
            (N, C_in) 逐点特征，例如 RGB 或 RGB+Normal。
        xyz   : torch.Tensor
            (N, 3) 逐点浮点坐标。
        batch : torch.Tensor
            (N,) 逐点 batch id，例如 [0,0,...,1,1,...,B,...]。

        返回
        ----
        logits : torch.Tensor
            (N, num_classes) 逐点分类 logits
        """
        # 1. 获取当前设备 逐点特征
        self.device = feats.device

        # 2. 构造稀疏坐标张量 (batch_id, x, y, z)
        #    coords 形状 (N, 4)
        coords = torch.cat([batch.unsqueeze(-1), xyz], dim=-1)

        # 3. 根据 signal 标志决定 sp 的输入特征
        if self.signal:
            # 如果 feats 包含除坐标外的额外特征（如颜色/法向量）
            if feats.shape[1] > 3:
                # 可选：把 feats 的最后 3 维替换为小数偏移量 (xyz - int(xyz))
                # 这样网络能感知子体素级别的精细位置
                if self.use_offset:
                    feats[:, -3:] = xyz - xyz.int()
            # 构造稀疏张量，特征用 feats，坐标用 coords
            sp = SparseTensor(feats.float(), coords.int(), device=self.device)
        else:
            # signal=False 时，忽略 feats，仅用全 1 作为占位特征
            sp = SparseTensor(
                torch.ones_like(feats).float(), coords.int(), device=self.device
            )

        # 4. 提取并归一化颜色（前 3 维）
        #    除以 1.001 是为了把 [0,1] 映射到略小于 1，防止整数溢出或量化误差
        colors = feats[:, 0:3] / 1.001

        # 5. 构造几何-颜色增强张量 coords_sp
        #    把坐标 (batch, x, y, z) 与颜色 (r, g, b) 拼接成 (N, 7)
        #    与 sp 共享坐标管理器，确保坐标顺序一致
        coords_sp = SparseTensor(
            features=torch.cat([coords, colors], dim=1),
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )

        # 6. 送入骨干网络，得到最终逐点 logits
        return self.backbone(sp, coords_sp)

    # def forward(self, feats, xyz, batch):
    #     self.device = feats.device
    #     coords = torch.cat([batch.unsqueeze(-1), xyz], dim=-1)
    #     if self.signal:
    #         if feats.shape[1] > 3:
    #             if self.use_offset:
    #                 feats[:, -3:] = xyz - xyz.int()
    #         sp = SparseTensor(feats.float(), coords.int(), device=self.device)
    #     else:
    #         sp = SparseTensor(
    #             torch.ones_like(feats).float(), coords.int(), device=self.device
    #         )
    #     colors = feats[:, 0:3] / 1.001
    #     coords_sp = SparseTensor(
    #         features=torch.cat([coords, colors], dim=1),
    #         coordinate_map_key=sp.coordinate_map_key,
    #         coordinate_manager=sp.coordinate_manager,
    #     )
    #
    #     return self.backbone(sp, coords_sp)
