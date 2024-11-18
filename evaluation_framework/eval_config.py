from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

class EvalConfig(BaseModel):
    """Evaluation configuration"""
    max_concurrent: int = Field(default=1, description="Maximum concurrent evaluations")
    throttle_value: int = Field(default=30, description="API throttling value")
    paths: Dict[str, Path] = Field(
        default={
            "test_cases": Path("eval_data/test_cases.json"),
            "results": Path("eval_data/results.json"),
            "custom_metrics": Path("eval_data/custom_metrics.json"),
            "synthetic_data": Path("synthetic_data")
        }
    )
    doc_retrieval_chunk_config: Dict[str, Any] = Field(
        default={
            "top_k": 1,
            "chunk_size": 150,
            "chunk_overlap": 50,
            "max_contexts": 1
        }
    )
    system_rag_config: Dict[str, Any] = Field(
        default={
            "top_k": 1,
        }
    )
    metrics_config: Dict[str, float] = Field(
        default={
            "contextual_precision_threshold": 0.5,
            "contextual_recall_threshold": 0.5,
            "contextual_relevancy_threshold": 0.5,
            "answer_relevancy_threshold": 0.5,
            "faithfulness_threshold": 0.5
        }
    )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "EvalConfig":
        """Load config from file"""
        if not config_path.exists():
            return cls()
        return cls.model_validate_json(config_path.read_text())