import timm
from pysrc import IndexAttention

def replace_qkv(model, attn_type, *args, **kwargs):
    # 遍历模型，找到 Attention 层并替换
    for name, module in model.named_modules():
        if isinstance(module, timm.models.vision_transformer.Attention):
            # print(f"Replacing Attention layer: {name} {module}")
            embed_dim = module.qkv.in_features  # 获取输入嵌入维度
            num_heads = module.num_heads  # 获取头的数量

            # 创建自定义的 Attention 层
            # 提取原始 Attention 模块的权重和偏置
            qkv_weight = module.qkv.weight.data.clone()
            qkv_bias = module.qkv.bias.data.clone()
            proj_weight = module.proj.weight.data.clone()
            proj_bias = module.proj.bias.data.clone()

            # 创建自定义的 Attention 层
            try:
                attention_class = globals()[attn_type]
                custom_attention = attention_class(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
            except KeyError:
                raise ValueError(f"Unsupported attention type: {attn_type}")

            # 将原始权重和偏置赋值给自定义的 Attention 层
            custom_attention.qkv.weight.data = qkv_weight
            custom_attention.qkv.bias.data = qkv_bias
            custom_attention.proj.weight.data = proj_weight
            custom_attention.proj.bias.data = proj_bias

            custom_attention.attn_drop.p = module.attn_drop.p
            custom_attention.proj_drop.p = module.proj_drop.p

            # 获取父模块
            parent_name = name.rsplit('.', 1)[0]
            parent_module = dict(model.named_modules())[parent_name]

            # 替换 Attention 层
            setattr(parent_module, name.split('.')[-1], custom_attention)

