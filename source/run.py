import sys, json

from argparse import ArgumentParser
from pathlib import Path

from .config import get_config
from .setup import build


def parse_arguments():
    parser = ArgumentParser("Training script.")

    parser.add_argument(
        "-s", "--source", 
        type=str, required=True
    )

    parser.add_argument(
        "-o", "--output",
        type=str, required=False,
        default="./output/" # + current_time_str()
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int, required=False,
        default=250
    )

    parser.add_argument(
        "-bs", "--batch_size",
        type=int, required=False,
        default=None
    )

    parser.add_argument(
        "--eval_interval",
        type=int, default=10
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true"
    )

    # Currently, this does only print config when starting.
    parser.add_argument(
        "-dv", "--devmode",
        action="store_true"
    )

    parser.add_argument(
        "-da", "--detect_anomaly",
        action="store_true"
    )
    
    return parser.parse_args(sys.argv[1:])

def process_arguments(args):
    def _process_source(s: str):
        image_path = Path(s)

        assert image_path.is_file(), \
            f"File not found: {image_path}"
        assert image_path.suffix.lower() in { ".png", ".jpg", ".jpeg" }, \
            f"Not a valid image file; expected a .png / .jpg / .jpeg file"

        name = image_path.parent.name
        return image_path, name

    def _process_out(o: str, name: str):
        output_path = Path(o) / name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _get_integer_list(max_iters: int, interval: int):
        return (
            list() if interval is None else 
            list(range(1, max_iters + 1, interval))
        )
    
    if args.detect_anomaly:
        import torch ; torch.autograd.set_detect_anomaly(True)

    config = get_config()

    if args.devmode: 
        print(json.dumps(config, indent=2))
    
    image_path, name = _process_source(args.source)
    log_dir = f"./logs/{name}"

    build_args = {
        "image_path": image_path, 
        "log_dir":    log_dir, 
        "config":     config
    }

    train_args = {
        "eval_iterations": _get_integer_list(args.iterations, args.eval_interval),
        "output_path":     _process_out(args.output, name),
        "batch_size":      args.batch_size, 
        "num_iterations":  args.iterations,
    }

    return build_args, train_args


def main():
    args = parse_arguments()
    build_args, train_args = process_arguments(args)

    build(**build_args).start(**train_args)


if __name__ == "__main__":
    main()
    