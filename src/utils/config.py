from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import os

def validate_range(value: float, min_val: float, max_val: float, field_name: str):
    if not (min_val <= value <= max_val):
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}, got {value}")

def validate_positive(value: float, field_name: str):
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")

@dataclass
class ConvolutiveConfig:
    filters: int
    frequency_bands: int
    frequency_range: Dict[str, int]
    bandwidth_range: Dict[str, int]
    coefficients: Dict[str, int]
    gain: Dict[str, int]
    bias: Dict[str, float]
    
    def __post_init__(self):
        validate_positive(self.filters, "filters")
        validate_range(self.filters, 1, 100, "filters")
        
        validate_positive(self.frequency_bands, "frequency_bands")
        validate_range(self.frequency_bands, 1, 20, "frequency_bands")
        
        validate_range(self.frequency_range['min'], 20, 20000, "frequency_range.min")
        validate_range(self.frequency_range['max'], 20, 20000, "frequency_range.max")
        if self.frequency_range['min'] > self.frequency_range['max']:
            raise ValueError("frequency_range.min must be less than or equal to frequency_range.max")
        
        validate_range(self.bandwidth_range['min'], 10, 5000, "bandwidth_range.min")
        validate_range(self.bandwidth_range['max'], 10, 5000, "bandwidth_range.max")
        if self.bandwidth_range['min'] > self.bandwidth_range['max']:
            raise ValueError("bandwidth_range.min must be less than or equal to bandwidth_range.max")
        
        validate_range(self.coefficients['min'], 0, 100, "coefficients.min")
        validate_range(self.coefficients['max'], 0, 100, "coefficients.max")
        if self.coefficients['min'] > self.coefficients['max']:
            raise ValueError("coefficients.min must be less than or equal to coefficients.max")
        
        validate_range(self.gain['min'], -50, 50, "gain.min")
        validate_range(self.gain['max'], -50, 50, "gain.max")
        if self.gain['min'] > self.gain['max']:
            raise ValueError("gain.min must be less than or equal to gain.max")
        
        validate_range(self.bias['min'], 0.0, 100.0, "bias.min")
        validate_range(self.bias['max'], 0.0, 100.0, "bias.max")
        if self.bias['min'] > self.bias['max']:
            raise ValueError("bias.min must be less than or equal to bias.max")

@dataclass
class ISDNoiseConfig:
    probability: int
    std_dev: float
    
    def __post_init__(self):
        validate_range(self.probability, 0, 100, "probability")
        validate_range(self.std_dev, 0.1, 10.0, "std_dev")

@dataclass
class SSINoiseConfig:
    snr_range: Dict[str, float]
    
    def __post_init__(self):
        validate_range(self.snr_range['min'], 0.0, 50.0, "snr_range.min")
        validate_range(self.snr_range['max'], 0.0, 50.0, "snr_range.max")
        if self.snr_range['min'] > self.snr_range['max']:
            raise ValueError("snr_range.min must be less than or equal to snr_range.max")

@dataclass
class TrainingConfig:
    # Dataset
    database_path: str
    protocols_path: str
    keys_path: str
    wav2vec_path: str
    
    # Base
    pretrained_model_path: Optional[str] = None
    model_tag: Optional[str] = None
    train: bool = True
    train_track: str = "LA"
    eval_track: str = "In-the-Wild"
    
    # Train
    batch_size: int = 10
    train_epoch: int = 200
    impove_epoch_patience: int = 10
    lr: float = 0.000001
    weight_decay: float = 0.0001
    loss: str = "WCE"
    algo: int = 5
    
    # RawBoost configurations
    convolutive: ConvolutiveConfig = field(default_factory=lambda: ConvolutiveConfig(
        filters=5,
        frequency_bands=5,
        frequency_range={'min': 20, 'max': 8000},
        bandwidth_range={'min': 100, 'max': 1000},
        coefficients={'min': 0, 'max': 100},
        gain={'min': -20, 'max': -5},
        bias={'min': 10, 'max': 100}
    ))
    
    isd_noise: ISDNoiseConfig = field(default_factory=lambda: ISDNoiseConfig(
        probability_range={'min': 0, 'max': 100},
        std_dev=2.0
    ))
    
    ssi_noise: SSINoiseConfig = field(default_factory=lambda: SSINoiseConfig(
        snr_range={'min': 10.0, 'max': 40.0}
    ))
    
    algorithm_combinations: Dict[int, List[str]] = field(default_factory=lambda: {
        4: ["convolutive", "isd_noise", "ssi_noise"],
        5: ["convolutive", "isd_noise"],
        6: ["convolutive", "ssi_noise"],
        7: ["isd_noise", "ssi_noise"],
        8: ["convolutive", "isd_noise"]
    })
    
    # Model
    emb_size: int = 144
    num_encoders: int = 12
    FT_W2V: bool = True
    seed: int = 1234
    comment: Optional[str] = None
    comment_eval: Optional[str] = None
    
    # Evaluation
    n_mejores_loss: int = 5
    average_model: bool = True
    n_average_model: int = 5
    
    def __post_init__(self):
        validate_range(self.lr, 1e-8, 1e-2, "lr")

        validate_range(self.weight_decay, 0.0, 1e-1, "weight_decay")
        
        validate_positive(self.batch_size, "batch_size")
        validate_range(self.batch_size, 1, 512, "batch_size")
        
        validate_positive(self.train_epoch, "train_epoch")
        validate_range(self.train_epoch, 1, 10000, "train_epoch")
        
        validate_range(self.algo, 0, 8, "algo")
        
        validate_positive(self.emb_size, "emb_size")
        validate_range(self.emb_size, 8, 1024, "emb_size")
        
        validate_positive(self.num_encoders, "num_encoders")
        validate_range(self.num_encoders, 1, 100, "num_encoders")
        
        valid_losses = ["WCE", "CE"]
        if self.loss not in valid_losses:
            raise ValueError(f"loss must be one of {valid_losses}, got {self.loss}")
        
        valid_train_tracks = ["LA"]
        if self.train_track not in valid_train_tracks:
            raise ValueError(f"train_track must be one of {valid_train_tracks}, got {self.train_track}")
        
        valid_val_tracks = ["LA", "DF", "In-the-Wild"]
        if self.eval_track not in valid_val_tracks:
            raise ValueError(f"eval_track must be one of {valid_val_tracks}, got {self.eval_track}")
    
    @classmethod
    def from_yaml(cls, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file does not exist: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if 'convolutive' in config_dict:
            config_dict['convolutive'] = ConvolutiveConfig(**config_dict['convolutive'])
        
        if 'isd_noise' in config_dict:
            config_dict['isd_noise'] = ISDNoiseConfig(**config_dict['isd_noise'])
        
        if 'ssi_noise' in config_dict:
            config_dict['ssi_noise'] = SSINoiseConfig(**config_dict['ssi_noise'])
        
        return cls(**config_dict)
    
    def save_to_yaml(self, config_path: str):
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        return config_dict
    
    def get_rawboost_params_for_algorithm(self) -> Dict[str, Any]:
        if self.algo not in self.algorithm_combinations:
            raise ValueError(f"unspport rawboost algorithm: {self.algo}")
        
        components = self.algorithm_combinations[self.algo]
        params = {}
        
        for component in components:
            if component == 'convolutive':
                params['convolutive'] = self.convolutive.__dict__
            elif component == 'isd_noise':
                params['isd_noise'] = self.isd_noise.__dict__
            elif component == 'ssi_noise':
                params['ssi_noise'] = self.ssi_noise.__dict__
        
        return params
    
    def __str__(self) -> str:
        lines = ["TrainingConfig:"]
    
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                lines.append(f"  {field_name}:")
                for nested_field, nested_value in field_value.__dict__.items():
                    lines.append(f"    {nested_field}: {nested_value}")
            elif isinstance(field_value, dict):
                lines.append(f"  {field_name}:")
                for key, value in field_value.items():
                    lines.append(f"    {key}: {value}")
            else:
                lines.append(f"  {field_name}: {field_value}")
        
        return "\n".join(lines)
    
    
def get_rawboost_args(config):
    return type('Args', (), {
        'N_f': config.convolutive.filters,
        'nBands': config.convolutive.frequency_bands,
        'minF': config.convolutive.frequency_range['min'],
        'maxF': config.convolutive.frequency_range['max'],
        'minBW': config.convolutive.bandwidth_range['min'],
        'maxBW': config.convolutive.bandwidth_range['max'],
        'minCoeff': config.convolutive.coefficients['min'],
        'maxCoeff': config.convolutive.coefficients['max'],
        'minG': config.convolutive.gain['min'],
        'maxG': config.convolutive.gain['max'],
        'minBiasLinNonLin': config.convolutive.bias['min'],
        'maxBiasLinNonLin': config.convolutive.bias['max'],
        'P': config.isd_noise.probability,
        'g_sd': config.isd_noise.std_dev,
        'SNRmin': config.ssi_noise.snr_range['min'],
        'SNRmax': config.ssi_noise.snr_range['max'],
    })()