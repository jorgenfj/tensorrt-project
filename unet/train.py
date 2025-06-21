import argparse
from roboflow import Roboflow
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--simple', action='store_true', default=False, help='Use a smaller UNet')
    parser.add_argument('--roboflow-api-key', type=str, default=None,)
    return parser.parse_args()

def main():
    args = get_args()
    project_root = Path.cwd()
    data_dir_base = project_root / "data"
    roboflow_api_key = args.roboflow_api_key
    if roboflow_api_key is None:
        raise ValueError("Roboflow API key must be provided with --roboflow-api-key argument")
    rf = Roboflow(roboflow_api_key) # api_key="Bc3tBeLXd35djgb8djKN"
    project = rf.workspace("pipe-92at4").project("pipeline-segmentation-nearby")
    version = project.version(11)
    dataset = version.download("png-mask-semantic", location=str(data_dir_base))