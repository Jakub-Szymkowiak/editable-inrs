from pathlib import Path

from ..train import Trainer

from .dataset   import RGBImageDataset
from .logger    import Logger

from .factories import build_model, build_optimizer


def build(
        image_path: Path,
        log_dir: str,
        config: dict,
    ):

    dataset = RGBImageDataset.from_path(image_path)
    logger = Logger(log_dir)
    
    model = build_model(config.get("model")).to(config.get("device"))
    optimizer = build_optimizer(model, config.get("optimizer"))

    return Trainer(
        dataset=dataset, 
        model=model, 
        optimizer=optimizer, 
        logger=logger
    )

    


