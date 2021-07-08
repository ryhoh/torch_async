from pytorch_pretrained_vit import ViT
from rotational_update import RotationalLinear


def apply_rotational_into_ViT_Projections(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in range(len(blocks)):
        attn = blocks[i].attn
        attn.proj_q = RotationalLinear(attn.proj_q, reduce_backward=False)
        attn.proj_k = RotationalLinear(attn.proj_k, reduce_backward=False)
        attn.proj_v = RotationalLinear(attn.proj_v, reduce_backward=False)
        blocks[i].proj = RotationalLinear(blocks[i].proj, reduce_backward=False)
    return model


def apply_rotational_into_ViT_PositionWiseFeedForward(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in range(len(blocks)):
        pwff = blocks[i].pwff
        pwff.fc1 = RotationalLinear(pwff.fc1, reduce_backward=False)
        pwff.fc2 = RotationalLinear(pwff.fc2, reduce_backward=False)
    return model
