import torch

def get_config():
    return {
        "device": "cuda",
        "model": {
            "anchors": {
                "type": "gaussian",
                "resolution": 128,
                "aspect_ratio": 16/9,
            },
            "bridge": {
                "k": 16,
                "chunk_size": 1024,
            },
            "hashgrid": {
                "n_input_dims": 2,
                "n_levels": 16,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 32,
                "per_level_scale": 1.5,
            },
            "decoder": {
                "out_dim": 3,
                "hidden_width": 256,
                "num_layers": 2,
                "activation": torch.nn.SiLU,
                "final_activation": None,
            },
        },
        "optimizer": {
            "name": "Adam",
            "args": { "betas": (0.9, 0.999) },
            "mode": "mul",
            "groups": {
                "anchors_pos": { "lr": 1e-2, "rule": {"name": "cosine", "args": {"value_start": 1.0, "value_end": 0.2, "steps": 5000}}},
                "anchors_scl": { "lr": 3e-7, "rule": {"name": "cosine", "args": {"value_start": 1.0, "value_end": 0.2, "steps": 5000}}},
                "anchors_ang": { "lr": 1e-3, "rule": {"name": "cosine", "args": {"value_start": 1.0, "value_end": 0.2, "steps": 5000}}},
                "decoder":     { "lr": 7e-3, "rule": {"name": "cosine", "args": {"value_start": 1.0, "value_end": 0.05, "steps": 5000}}},
                "hashgrid":    { "lr": 7e-3, "rule": {"name": "cosine", "args": {"value_start": 1.0, "value_end": 0.2,  "steps": 5000}}}
            }
        }
    }