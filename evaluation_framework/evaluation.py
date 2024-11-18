"""
LLM Evaluation Script using deepeval metrics.
Evaluates LLM responses for various quality metrics including context precision,
recall, relevancy, faithfulness, and legal risk assessment.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from quart import Quart
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from models import ConfigLoader, ModelFactory
from system_rag import RAG

load_dotenv()

class RAGEvaluator:
    """Handles evaluation of RAG system responses"""
    
    def __init__(self,
                 config,
                 app: Optional[Quart] = None):
        """
        Initialize the Evaluator with the provided configuration and evaluation data.
        """
        self.config = config
        self.results_path = config.paths["results"]
        self.custom_metrics = config.paths["custom_metrics"]
        self.max_concurrent = config.max_concurrent
        self.throttle_value = config.throttle_value
        self.app = app
        # Initialize configuration
        if self.app:
            self.llm_model = create_model_from_app_config(self.app)
        else:
            self.llm_model = create_llm_model()

    async def process_goldens(self, goldens: List[Dict], rag_system: RAG) -> List[Dict]:
        """Process goldens and return evaluation data"""
        # Process all questions in one batch
        goldens = [g.model_dump() for g in goldens]

        eval_data_list = []
        questions = []
        for golden in goldens:
            eval_data = {
                "input": golden["input"],
                "expected_output": golden["expected_output"],
                "context": golden["context"]
            }
            eval_data_list.append(eval_data)
            questions.append(eval_data["input"])

        answers = await rag_system.ask_questions(questions)

        # Update eval_data_list with results
        for i, eval_data in enumerate(eval_data_list):
            _answer = answers["results"][i]["message"]["content"]
            _context = answers["results"][i]["context"]['data_points']["text"]
            eval_data["actual_output"] = _answer
            eval_data["retrieval_context"] = _context

        return eval_data_list
    
    def read_test_cases(self) -> List[LLMTestCase]:
        """Read test cases from the provided file"""
        if self.config.paths["test_cases"]:
            return read_test_cases(self.config.paths["test_cases"])
        return []
    
    async def _evaluate(self, eval_data: Optional[List[Dict]] = None) -> None:
        """
        Run evaluation on the provided test cases.
        eval_data: List of evaluation data
        """
        test_cases = self.read_test_cases()

        if not eval_data and not test_cases:
            raise ValueError("Please provide test cases or a path to a test cases file.")

        custom_metrics = read_json(self.custom_metrics)
        metrics = create_metrics(self.llm_model, custom_metrics)

        # Read test cases from the provided eval_data list
        if eval_data:
            test_cases_data = [LLMTestCase(**test_case) for test_case in eval_data]
            test_cases += test_cases_data
        # Run evaluation
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            throttle_value=self.throttle_value,
            max_concurrent=self.max_concurrent
        )
        # Save results
        if self.results_path:
            save_results(results, self.results_path)


def create_llm_model() -> DeepEvalBaseLLM:
    llm_config = ConfigLoader.load_llm_config()
    llm_model = ModelFactory.create_llm_model(llm_config)
    return llm_model

def create_model_from_app_config(current_app):
    llm_model, _ = ModelFactory.from_app_config(current_app)
    return llm_model

def create_metrics(eval_model: DeepEvalBaseLLM,
                   custom_metrics: Optional[Dict]=[]) -> List:
    """Create a list of evaluation metrics."""
    # Create custom metrics
    metrics = []
    for metric in custom_metrics["metrics"]:
        metric_object = GEval(
            model=eval_model,
            name=metric["name"],
            criteria=metric["description"],
            evaluation_params=[LLMTestCaseParams.INPUT,
                               LLMTestCaseParams.ACTUAL_OUTPUT]
        )
        metrics.append(metric_object)

    return [
        ContextualPrecisionMetric(model=eval_model, include_reason=False),
        ContextualRecallMetric(model=eval_model, include_reason=False),
        ContextualRelevancyMetric(model=eval_model, include_reason=False),
        AnswerRelevancyMetric(model=eval_model, include_reason=False),
        FaithfulnessMetric(model=eval_model, include_reason=False),
    ] + metrics

def read_json(filename: str) -> dict:
    if filename:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        return data
    return {}

def read_test_cases(filename: str) -> List[LLMTestCase]:
    """Read test cases from a JSON file.
    test_cases.json should be a list of dictionaries, each containing the
    following keys:
    - input: str
    - actual_output: str
    - expected_output: str
    - retrieval_context: List[str]
    """
    test_cases = read_json(filename)["test_cases"]
    
    return [LLMTestCase(**test_case) for test_case in test_cases]


def save_results(results: dict, filename: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results.model_dump(), f, indent=4)