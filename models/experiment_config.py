from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

class ContaminationType(Enum):
    ARSENIC = "arsenic"
    CHLORINE = "chlorine"

@dataclass
class ExperimentConfig:
    model_name: str
    config_name: str
    window_size: int = 30
    desinfectant: ContaminationType = ContaminationType.CHLORINE
    contaminated_files: List[str] = field(default_factory=list) # data files (used for train and test if no example file)
    nodes: List[str] = field(default_factory=list)          # list of nodes on which models are trained/tested
    contaminants: List[ContaminationType] = field(default_factory=lambda: [ContaminationType.ARSENIC])                  
    model_params: Optional[Dict] = None
    example_files: Optional[List[str]] = field(default_factory=list) # example files of normal behavior if need of examples of clean data
    aggregate_method: Optional[str] = None    # whether models are trained on each node separately or on aggregated nodes (e.g., mean/sum)
    

    