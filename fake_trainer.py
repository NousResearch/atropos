#!/usr/bin/env python3
"""
Simple fake trainer that registers with the API server and fetches batches
to allow environments to generate data when using dump_rollouts mode.
"""
import asyncio
import json
import logging
import time
import requests
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeTrainer:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.trainer_uuid: Optional[str] = None
        
    def register(self) -> bool:
        """Register as a trainer with the API server"""
        registration_data = {
            "wandb_group": "fake_trainer",
            "wandb_project": "fake_trainer", 
            "batch_size": 16,
            "max_token_len": 16384,
            "starting_step": 0,
            "checkpoint_dir": "/tmp/fake_trainer",
            "save_checkpoint_interval": 1000,
            "num_steps": 50000,
            "group_size": 16,
            "model_name": "fake-model",
            "world_size": 1,
            "tokenizer_name": "deepseek-ai/DeepSeek-R1",
            "trainer_type": "FAKE_TRAINER"
        }
        
        try:
            response = requests.post(f"{self.api_url}/register", json=registration_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                self.trainer_uuid = result.get('uuid')
                logger.info(f"Successfully registered as trainer with UUID: {self.trainer_uuid}")
                return True
            else:
                logger.error(f"Failed to register: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error registering trainer: {e}")
            return False
    
    def fetch_batch(self) -> Optional[dict]:
        """Fetch a batch from the API server"""
        try:
            response = requests.get(f"{self.api_url}/batch", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch batch: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            return None
    
    def run(self):
        """Main loop: continuously fetch and discard batches"""
        if not self.register():
            logger.error("Failed to register trainer, exiting")
            return
            
        logger.info("Starting fake trainer loop - fetching batches to allow environments to proceed")
        
        batches_processed = 0
        empty_batches = 0
        
        while True:
            try:
                # Fetch batch
                data = self.fetch_batch()
                
                if data and data.get("batch") is not None:
                    batches_processed += 1
                    batch_size = len(data["batch"])
                    logger.info(f"Fetched batch {batches_processed} with {batch_size} items")
                    empty_batches = 0  # Reset empty batch counter
                else:
                    empty_batches += 1
                    if empty_batches % 10 == 0:  # Log every 10 empty batches
                        logger.info(f"No batch available (empty batches: {empty_batches})")
                    time.sleep(1)  # Wait before trying again
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait longer on errors

if __name__ == "__main__":
    trainer = FakeTrainer()
    trainer.run()