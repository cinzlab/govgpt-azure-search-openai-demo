import asyncio
import logging
from pathlib import Path

import pytest
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import assert_test
from deepeval.test_case import LLMTestCase

from evaluation import RAGEvaluator
from eval_config import EvalConfig
from system_rag import create_RAG_eval_app, RAG

logger = logging.getLogger(__name__)

async def prepare_models(config: EvalConfig):
    """Prepare RAG system and evaluator"""
    app = await create_RAG_eval_app()
    evaluator = RAGEvaluator(config, app=app)
    RAG_system = RAG(app)
    return RAG_system, evaluator

async def run_evaluation():
    """Run the complete evaluation process"""
    # Initialize models
    config = EvalConfig.from_file(Path("eval_config.json"))
    RAG_system, evaluator = await prepare_models(config)
    # Load test data once
    test_data_rag = evaluator.load_test_cases(config.paths["test_cases_rag"])
    test_data_dummy = evaluator.load_test_cases(config.paths["test_cases_dummy"])
    # Create dummy test cases (for expected behavior)
    dummy_test_cases, metrics = evaluator.prepare_tests(test_data_dummy)
    # Create RAG-answered test cases (actual system responses)
    answered_test_data = await evaluator.prepare_eval_data(test_data_rag, RAG_system)
    answered_test_cases, _ = evaluator.prepare_tests(answered_test_data)
    
    return dummy_test_cases, answered_test_cases, metrics

# Run async setup once at module level
dummy_test_cases, answered_test_cases, metrics = asyncio.run(run_evaluation())

# Create datasets for testing
dummy_dataset = EvaluationDataset(test_cases=dummy_test_cases)
rag_dataset = EvaluationDataset(test_cases=answered_test_cases)

@pytest.mark.parametrize("test_case", dummy_dataset)
def test_dummy_responses(test_case: LLMTestCase):
    """Test expected responses using prepared metrics
    
    This test validates that our expected outputs meet our quality criteria.
    It uses the pre-defined answers from our test cases file.
    """
    assert_test(test_case=test_case, metrics=metrics)

@pytest.mark.parametrize("test_case", rag_dataset)
def test_rag_responses(test_case: LLMTestCase):
    """Test actual RAG responses using prepared metrics
    
    This test validates that our RAG system's actual responses meet our quality criteria.
    It uses real responses generated by running queries through the RAG system.
    """
    assert_test(test_case=test_case, metrics=metrics)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])