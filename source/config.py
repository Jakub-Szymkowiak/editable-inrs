import torch
# import train.losses as Ls   # uncomment when you want to hook up losses


def get_config():
    return {
        "device": "cuda",

        "densification": {
            "prune_int":     10,
            "densf_int":     10,
            "start_at_iter": 0,
            "end_at_iter":   3_500,
        }, 

        "eval": {
            "int": 250, "start": 0, "end": 10_000
        },

        "model": {
            "anchors": {
                "type":         "gaussian",
                "resolution":   256,
                "aspect_ratio": 16 / 9,
            },
            "bridge": {
                "k":          12,
                "chunk_size": 4096,
            },
            "hashgrid": {
                "n_input_dims":        2,
                "n_levels":            16,
                "n_features_per_level": 8,
                "log2_hashmap_size":   19,
                "base_resolution":     128,
                "per_level_scale":     1.5,
            },
            "decoder": {
                "out_dim":          3,
                "hidden_width":     256,
                "num_layers":       2,
                "activation":       torch.nn.SiLU,
                "final_activation": None,
            },
            "trimmer": {
                "type":       "GaussianTrimmer",
                "ema_beta":   0.95,
                "prune_thrs": 5e-6,
                "prune_cap":  0.10,
                "densf_thrs": 0.5,
                "jitter":     1e-2
            },
        },

        # --- optimizer spec ---
        "optimizer": {
            "name": "Adam",
            "args": {"betas": (0.9, 0.999)},
            "groups": {
                "anchors_pos": {
                    "lr": 1e-1,
                    "rule": {"name": "cosine", "args": {
                        "value_start": 1.0, "value_end": 0.1, "steps": 5000
                    }},
                },
                "anchors_scl": {
                    "lr": 3e-5,
                    "rule": {"name": "cosine", "args": {
                        "value_start": 1.0, "value_end": 0.1, "steps": 5000
                    }},
                },
                "anchors_ang": {
                    "lr": 1e-3,
                    "rule": {"name": "cosine", "args": {
                        "value_start": 1.0, "value_end": 0.1, "steps": 5000
                    }},
                },
                "decoder": {
                    "lr": 7e-3,
                    "rule": {"name": "cosine", "args": {
                        "value_start": 1.0, "value_end": 0.05, "steps": 5000
                    }},
                },
                "hashgrid": {
                    "lr": 7e-3,
                    "rule": {"name": "cosine", "args": {
                        "value_start": 1.0, "value_end": 0.1, "steps": 5000
                    }},
                },
            },
        },

        "losses": {
            "l1": {
                "fn":   { "name": "L1Loss",   "args": {} },
                "rule": { "name": "constant", "args": { "value": 1.0 } }
            },
            # "ssim": {
            #     "fn":   { "name": "FusedSSIM", "args": { "window": 11 } },
            #     "rule": { "name": "cosine",    "args": { "value_start": 0.0, "value_end": 0.3, "steps": 3000 } }
            # }
        }
    }