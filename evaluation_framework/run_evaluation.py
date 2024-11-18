from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import logging

from evaluation import RAGEvaluator
from eval_config import EvalConfig
from system_rag import RAG, create_RAG_eval_app, cleanup_clients
from golden_generation import synthetaze_data


logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class EvaluationPipeline:
    """Evaluation pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = EvalConfig.from_file(config_path or Path("eval_config.json"))
        
    async def run(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        app = await create_RAG_eval_app()
        
        try:
            logger.info("Starting evaluation pipeline")

            RAG_system = RAG(app)
            contexts = await RAG_system.get_contexts(top=self.config.system_rag_config["top_k"])
            context_texts = [[ctx.content] for ctx in contexts if ctx.content]

            evaluator = RAGEvaluator(self.config, app=app)
            
            goldens = await synthetaze_data(documents=context_texts,
                                            max_contexts=self.config.system_rag_config["top_k"],
                                            save_path=self.config.paths["synthetic_data"],
                                            app_config=app,
                                            gen_from_docs=False)
            
            context_eval_data = await evaluator.process_goldens(goldens, RAG_system)
            results = await evaluator._evaluate(context_eval_data)

            logger.info("Evaluation pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise ValueError(f"Pipeline failed: {str(e)}") from e
            
        finally:
            await cleanup_clients(app)

async def main():
    pipeline = EvaluationPipeline()
    results = await pipeline.run()
    print(results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())