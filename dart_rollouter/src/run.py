import ray
import asyncio
from pathlib import Path
import time
import copy
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Updated import paths based on new project structure
from src.core.task_loader import TaskLoader
from src.core.agent_coordinator import AgentCoordinator
from src.services.model_service_pool import ModelServicePool
from src.services.storage_actor import StorageActor
from src.services.mysql_writer import MySQLWriterActor
from src.utils.log_config import setup_logging
from src.utils.kill_all_env import clean_env

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)

# Set environment variables
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    asyncio.run(async_main(config))

def _init_ray():
    """Connect to RAY_ADDRESS if set; otherwise start/join the default local cluster."""
    address = os.environ.get("RAY_ADDRESS") or None
    namespace = os.environ.get("RAY_NAMESPACE", "dart")
    dashboard_port = os.environ.get("RAY_DASHBOARD_PORT")
    init_kwargs = {
        "log_to_driver": True,
        "namespace": namespace,
    }
    if dashboard_port:
        init_kwargs["dashboard_port"] = int(dashboard_port)
    if address:
        logger.info(f"Connecting Ray at {address} namespace={namespace}")
        ray.init(address=address, **init_kwargs)
    else:
        init_kwargs["dashboard_host"] = "0.0.0.0"
        logger.info(f"Starting/joining default local Ray namespace={namespace}")
        ray.init(**init_kwargs)


async def async_main(config=None):
    _init_ray()

    print(f">>>taskloader begin to load tasks")
    task_loader = TaskLoader(config.task, config.storage.root, config.coordinator.rollout_n)
    print(f">>>coordinator begin to init")
    coordinator = AgentCoordinator(config.coordinator)
    print(f">>>storage begin to init")
    print(f">>>config.storage:>>> {config.storage}")
    storage = StorageActor.remote(config.storage)
  
    mysql_writer = MySQLWriterActor.remote(config.mysql)
   
    model_pool = None
    model_pool = ModelServicePool.remote(model_cfg=config.model)

    process_reward_cfg = config.process_reward
    pr_backend = str(process_reward_cfg.backend).lower()
    if pr_backend not in {"disabled", "original", "seekjudge"}:
        raise ValueError(f"Unsupported process_reward.backend: {pr_backend}")
    if pr_backend == "disabled":
        pr_model_pool = None
    else:
        pr_model_pool = ModelServicePool.remote(model_cfg=process_reward_cfg.model)
    
    # Clear environment
    clean_env()

    while True:
        try:
            completed = await coordinator.start_rollout(
                task_loader=task_loader,
                runner_cfg=config.runner,
                model_pool=model_pool,
                storage=storage,
                mysql_writer=mysql_writer,
                pr_model_pool=pr_model_pool,
                process_reward_cfg=process_reward_cfg,
            )
            
            # Exit main loop if all tasks are completed
            if completed:
                logger.info("All tasks completed, program exits normally")
                break
                
        except KeyboardInterrupt:
            logger.info("Interrupt signal received, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error occurred in main loop: {e}")
            break

if __name__ == "__main__":
    main()
