from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class ExperimentConfig:
    model_name: str
    window_size: int = 30
    desinfectant: str = "chlorine" 
    contaminated_files: List[str] = field(default_factory=list) # data files (used for train and test if no example file)
    nodes: List[str] = field(default_factory=list)          # list of nodes on which models are trained/tested
    contaminants: List[str] = field(default_factory=lambda: ["arsenic"])                  
    model_params: Optional[Dict] = None
    example_files: Optional[List[str]] = field(default_factory=list) # example files of normal behavior if need of examples of clean data
    aggregate_method: Optional[str] = None    # whether models are trained on each node separately or on aggregated nodes (e.g., mean/sum)
    