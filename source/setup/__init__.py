from pathlib import Path

from ..train.losses import LossFunction
from ..train import Trainer

from .dataset   import RGBImageDataset
from .logger    import Logger
from. optimizer import Optimizer
from .schedule  import Schedule

from .factory import build_model


def build(
        image_path: Path,
        log_dir: str,
        config: dict,
    ):
    
    model = build_model(config.get("model")).to(config.get("device"))
    
    dataset   = RGBImageDataset.from_path(image_path)
    lossfunc  = LossFunction.from_config(config.get("losses"))
    optimizer = Optimizer.from_config(model, config.get("optimizer"))
    schedule  = Schedule.from_config(config)
    logger    = Logger(log_dir)

    return Trainer(model, dataset, lossfunc, optimizer, schedule, logger)

    


