import torch

from src.algorithms.classifier import Classifier
from src.algorithms.components.jetformer import JetFormerNet
from src.import_teacher import map_state_dict

ADAM = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}
JETFORMER = {
    "_target_": "src.algorithms.components.jetformer.JetFormerNet",
    "embed_dim": 16,
    "num_heads": 2,
    "num_transformers": 2,
}
# The keys of one block of a jet_tagging_gdl JetFormer checkpoint
# (pt_eta_phi/jetformer/nconst50/baseline), in its order.
SIBLING_BLOCK_KEYS = (
    "attn.norm.norm.weight",
    "attn.norm.norm.bias",
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.out.weight",
    "attn.out.bias",
    "norm_0.norm.weight",
    "norm_0.norm.bias",
    "linear_0.weight",
    "norm_1.norm.weight",
    "norm_1.norm.bias",
    "linear_1.weight",
)


def _sibling_keys(n_blocks: int) -> list[str]:
    stem = ["net.cls_token", "net.embedding.weight", "net.embedding.bias"]
    blocks = [
        f"net.blocks.{i}.{k}" for i in range(n_blocks) for k in SIBLING_BLOCK_KEYS
    ]
    norm = ["net.norm.weight", "net.norm.bias"]
    return [*stem, *blocks, *norm, "net.classifier.weight", "net.classifier.bias"]


def test_a_sibling_checkpoint_loads_strictly() -> None:
    module = Classifier(
        net=JETFORMER, num_classes=5, input_shape=(8, 3), optimizer=ADAM
    )
    state = {k: torch.randn_like(v) for k, v in module.state_dict().items()}

    module.load_state_dict(map_state_dict(state, [], {}), strict=True)

    assert list(state) == _sibling_keys(2)
    assert all(torch.equal(module.state_dict()[k], v) for k, v in state.items())


def test_the_hint_is_the_normed_cls_token_the_classifier_reads() -> None:
    torch.manual_seed(0)
    net = JetFormerNet(embed_dim=16, num_heads=2, num_transformers=2).eval()
    read = []
    net.classifier.register_forward_pre_hook(lambda module, args: read.append(args[0]))
    x = torch.randn(4, 8, 3)

    logits, hint = net.forward_with_hint(x)

    assert torch.equal(hint, read[0]) and hint.shape == (4, net.hint_dim)
    assert torch.equal(net(x), logits)
