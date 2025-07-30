"""
Celery tasks for asynchronous model training
"""

from celery import Task
from backend.celery_app import celery_app
from backend.ml.train import ModelTrainer
from backend.models.database import SessionLocal
import logging

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Task with database session management"""
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} succeeded with result: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed with exception: {exc}")


@celery_app.task(base=CallbackTask, name="train_all_models")
def train_all_models():
    """Train all ML models (GMM and Neural Networks)"""
    logger.info("Starting model training task...")
    
    try:
        trainer = ModelTrainer()
        
        # Train GMM clustering model
        logger.info("Training GMM clustering model...")
        gmm_result = trainer.train_gmm_model()
        
        # Train neural network models for each position
        logger.info("Training neural network models...")
        nn_results = {}
        for position in ["QB", "RB", "WR", "TE"]:
            nn_results[position] = trainer.train_position_model(position)
        
        result = {
            "status": "success",
            "gmm": gmm_result,
            "neural_networks": nn_results
        }
        
        logger.info("Model training completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


@celery_app.task(base=CallbackTask, name="train_position_model")
def train_position_model(position: str):
    """Train model for a specific position"""
    logger.info(f"Training model for position: {position}")
    
    try:
        trainer = ModelTrainer()
        result = trainer.train_position_model(position)
        
        logger.info(f"Model training for {position} completed")
        return result
        
    except Exception as e:
        logger.error(f"Model training for {position} failed: {str(e)}")
        raise


@celery_app.task(base=CallbackTask, name="evaluate_models")
def evaluate_models():
    """Evaluate all trained models"""
    logger.info("Starting model evaluation...")
    
    try:
        trainer = ModelTrainer()
        evaluation_results = trainer.evaluate_all_models()
        
        logger.info("Model evaluation completed")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise