import torch

def get_config():
    return {
        "device": "cuda",
        "model": {
            "anchors": {
                "resolution": 256,
                "aspect_ratio": 16/9,
            },
            "bridge": {
                "k": 32,
                "bandwidth": 0.07,
                "chunk_size": 4096,
            },
            "hashgrid": {
                "n_input_dims": 2,
                "n_levels": 16,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            "decoder": {
                "out_dim": 3,
                "hidden_width": 512,
                "num_layers": 3,
                "activation": torch.nn.SiLU,
                "final_activation": None,
            },
        },
        "optimizer": {
            "name": "Adam",
            "args": { "betas": (0.9, 0.999) },
            "mode": "mul",
            "groups": {
                "anchors": {
                    "lr": 5e-3,
                    "rule": {
                        "name": "cosine",
                        "args": { "value_start": 1.0,
                                  "value_end": 0.3,
                                  "steps": 2000 }
                    },
                },
                "decoder": {
                    "lr": 1e-2,
                    "rule": {
                        "name": "cosine",
                        "args": { "value_start": 1.0,
                                  "value_end": 0.1,
                                  "steps": 2000 }
                    },
                },
                "hashgrid": {
                    "lr": 1e-2,
                    "rule": {
                        "name": "constant",
                        "args": { "value": 1.0 }
                    },
                },
            },
        },
    }