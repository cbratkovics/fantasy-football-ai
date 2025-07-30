"""
Celery configuration for asynchronous task processing
"""

from celery import Celery
import os

# Get Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "fantasy_football",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "backend.tasks.train_models",
        "backend.tasks.update_data"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=3600,       # 1 hour
    # Result backend settings
    result_expires=86400,       # 24 hours
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
    # Beat schedule for periodic tasks
    beat_schedule={
        "update-player-data": {
            "task": "backend.tasks.update_data.update_player_data",
            "schedule": 86400.0,  # Daily
            "options": {"queue": "data_updates"}
        },
        "train-models-weekly": {
            "task": "backend.tasks.train_models.train_all_models",
            "schedule": 604800.0,  # Weekly
            "options": {"queue": "ml_training"}
        }
    },
    # Queue routing
    task_routes={
        "backend.tasks.train_models.*": {"queue": "ml_training"},
        "backend.tasks.update_data.*": {"queue": "data_updates"}
    }
)

if __name__ == "__main__":
    celery_app.start()