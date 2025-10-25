from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import os

@dataclass
class ConvolutiveConfig:
    filters: int
    frequency_bands: int
    frequency_range: Dict[str, int]
    bandwidth_range: Dict[str, int]
    coefficients: Dict[str, float]
    gain: Dict[str, float]
    bias: Dict[str, float]

@dataclass
class ISDNoiseConfig:
    probability: float
    std_dev: float

@dataclass
class SSINoiseConfig:
    snr_range: Dict[str, float]

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
        filters=10,
        frequency_bands=5,
        frequency_range={'min': 20, 'max': 8000},
        bandwidth_range={'min': 100, 'max': 1000},
        coefficients={'min': 0.1, 'max': 0.9},
        gain={'min': 0.1, 'max': 0.9},
        bias={'min': 0.1, 'max': 0.9}
    ))
    
    isd_noise: ISDNoiseConfig = field(default_factory=lambda: ISDNoiseConfig(
        probability=0.5,
        std_dev=2.0
    ))
    
    ssi_noise: SSINoiseConfig = field(default_factory=lambda: SSINoiseConfig(
        snr_range={'min': 10.0, 'max': 30.0}
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