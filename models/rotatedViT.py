from pytorch_pretrained_vit import ViT
from rotational_update import RotationalLinear

from layers.enhanced_rotational_linear import EnhancedRotationalLinear


def apply_rotational_into_ViT_Projections(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in range(len(blocks)):
        attn = blocks[i].attn
        attn.proj_q = EnhancedRotationalLinear(attn.proj_q, reduce_backward=False)
        attn.proj_k = EnhancedRotationalLinear(attn.proj_k, reduce_backward=False)
        attn.proj_v = EnhancedRotationalLinear(attn.proj_v, reduce_backward=False)
        blocks[i].proj = EnhancedRotationalLinear(blocks[i].proj, reduce_backward=False)
    return model


def apply_rotational_into_ViT_PositionWiseFeedForward(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in range(len(blocks)):
        pwff = blocks[i].pwff
        pwff.fc1 = EnhancedRotationalLinear(pwff.fc1, reduce_backward=False)
        pwff.fc2 = EnhancedRotationalLinear(pwff.fc2, reduce_backward=False)
    return model
