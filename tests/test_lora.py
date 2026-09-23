import torch
from safetensors.torch import load_file
from torch import nn

from yue2_lora.lora import LoRALinear, export_native_ar, fuse_pairs


def test_fuse_pairs_preserves_concatenated_delta() -> None:
    generator = torch.Generator().manual_seed(7)
    pairs = [
        (torch.randn(2, 5, generator=generator), torch.randn(3, 2, generator=generator)),
        (torch.randn(4, 5, generator=generator), torch.randn(6, 4, generator=generator)),
        (torch.randn(1, 5, generator=generator), torch.randn(2, 1, generator=generator)),
    ]
    down, up = fuse_pairs(pairs)
    expected = torch.cat([b @ a for a, b in pairs])
    torch.testing.assert_close(up @ down, expected)


def test_native_ar_export_uses_comfyui_yue2_clip_keys(tmp_path) -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = LoRALinear(nn.Linear(8, 8, bias=False), 2)
            self.k_proj = LoRALinear(nn.Linear(8, 4, bias=False), 2)
            self.v_proj = LoRALinear(nn.Linear(8, 4, bias=False), 2)
            self.o_proj = LoRALinear(nn.Linear(8, 8, bias=False), 2)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = LoRALinear(nn.Linear(8, 16, bias=False), 2)
            self.up_proj = LoRALinear(nn.Linear(8, 16, bias=False), 2)
            self.down_proj = LoRALinear(nn.Linear(16, 8, bias=False), 2)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

    model = nn.Module()
    model.config = type("Config", (), {"num_hidden_layers": 1})()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Layer()])
    path = tmp_path / "native.safetensors"
    export_native_ar(model, path, {"rank": 2})
    state = load_file(path)
    prefix = "text_encoders.model.layers.0."
    assert set(state) == {
        prefix + "self_attn.qkv_proj.lora_down.weight",
        prefix + "self_attn.qkv_proj.lora_up.weight",
        prefix + "self_attn.o_proj.lora_down.weight",
        prefix + "self_attn.o_proj.lora_up.weight",
        prefix + "mlp.gate_up_proj.lora_down.weight",
        prefix + "mlp.gate_up_proj.lora_up.weight",
        prefix + "mlp.down_proj.lora_down.weight",
        prefix + "mlp.down_proj.lora_up.weight",
    }
