from pytorch_pretrained_vit import ViT
from rotational_update import RotationalLinear


def apply_rotational_into_ViT_Projections(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in len(blocks):
        attn = blocks[i].attn
        attn.proj_q = RotationalLinear(attn.proj_q)
        attn.proj_k = RotationalLinear(attn.proj_k)
        attn.proj_v = RotationalLinear(attn.proj_v)
        blocks[i].proj = RotationalLinear(blocks[i].proj)
    return model


def apply_rotational_into_ViT_PositionWiseFeedForward(model: ViT) -> ViT:
    blocks = model.transformer.blocks
    for i in len(blocks):
        pwff = blocks[i].pwff
        pwff.fc1 = RotationalLinear(pwff.fc1)
        pwff.fc2 = RotationalLinear(pwff.fc2)
    return model
