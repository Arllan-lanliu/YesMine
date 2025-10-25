import argparse
import torch

from src.utils.config import TrainingConfig
from src.core.model import Model
from src.utils.util import reproducibility
from src.core.train import train_model
from src.core.eval import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'XLSR-Mamba')
    parser.add_argument('--config', 
                       type = str, 
                       required = True, 
                       default = './conf/base.yaml', 
                       help = 'yaml config path')
    parser.add_argument('--override', 
                       type = str, 
                       nargs = '+', 
                       help = 'override, e.g. --override batch_size=32 lr=0.0001')

    args = parser.parse_args()

    print("Load config")
    config = TrainingConfig.from_yaml(args.config)
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            setattr(config, key, value)
    print(config)

    # Make experiment reproducible
    reproducibility(config.seed, config)

    assert config.n_average_model < config.n_mejores_loss + 1, 'Average models must be smaller or equal to number of saved epochs'

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = Model(config, device)
    if not config.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    model = model.to(device)

    if not config.pretrained_model_path and config.train:
        train_model(config, model, device)

    evaluate(config, model, device)
    