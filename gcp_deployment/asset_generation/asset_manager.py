import uuid
import os
import logging # Added logging
from dataclasses import dataclass, field, asdict # Added asdict
from typing import Optional, List, Dict, Any # Added List, Dict, Any

from .point_e_handler import PointEHandler # Assuming point_e_handler is in the same directory

logger = logging.getLogger(__name__) # Added logger instance

@dataclass
class Asset:
    agent_id: str
    description: str
    file_path: str
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    complexity: str = "simple" # Default complexity

    def to_dict(self) -> Dict[str, Any]:
        # asset_id is already a string via uuid.uuid4()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        # Ensure all fields required by __init__ are present or provide defaults if necessary
        # The dataclass __init__ should handle this if data keys match field names.
        return cls(**data)

class AssetManager:
    def __init__(self, output_directory: str = "generated_assets"):
        self.point_e_handler = PointEHandler() # Consider making device configurable
        self.output_directory = output_directory
        try:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
                logger.info(f"AssetManager: Created output directory: {self.output_directory}")
            else:
                logger.info(f"AssetManager: Output directory already exists: {self.output_directory}")
        except OSError as e:
            logger.error(f"AssetManager: Error creating output directory {self.output_directory}: {e}", exc_info=True)
            # Depending on severity, could raise an error or default to a known safe path like /tmp
            # For now, it will just log, and asset saving might fail later if dir doesn't exist.
            # Consider setting a fallback or raising to prevent silent failures in asset saving.
            # self.output_directory = "/tmp/generated_assets_fallback" # Example fallback
            # raise e # Or re-raise if directory creation is critical for startup
        
        self.assets_created: List[Asset] = [] # A simple in-memory registry for now

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the AssetManager's dynamic state."""
        return {
            "assets_created": [asset.to_dict() for asset in self.assets_created]
            # output_directory is an init parameter, not typically part of dynamic state to save here
            # point_e_handler is an object that is re-initialized, not serialized directly
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], output_directory: Optional[str] = None) -> "AssetManager":
        """Deserializes state into a new AssetManager instance."""
        # If output_directory is not provided in data (recommended to pass it explicitly if different from default)
        # it will use the default from __init__.
        # For more flexibility, could pass output_directory from the global config during full simulation load.
        manager = cls(output_directory=output_directory or "generated_assets")
        
        assets_data = data.get("assets_created", [])
        manager.assets_created = [Asset.from_dict(asset_data) for asset_data in assets_data]
        
        logger.info(f"AssetManager state restored with {len(manager.assets_created)} assets.")
        return manager

    async def create_asset_for_agent(self, agent_id: str, description: str, complexity: str = "simple") -> Optional[Asset]:
        """
        Generates a 3D asset using PointEHandler based on the agent's description.
        Saves the asset and returns an Asset object.
        """
        logger.info(f"AssetManager: Received request from agent {agent_id} to create '{description}' (Complexity: {complexity}).")
        
        # Generate a unique filename
        asset_uuid = uuid.uuid4()
        file_name = f"agent_{agent_id}_asset_{asset_uuid}.ply"
        # Ensure output_directory is valid before joining; __init__ handles creation but could have failed silently before this change.
        if not os.path.isdir(self.output_directory): # Check if directory is usable
            logger.error(f"AssetManager: Output directory '{self.output_directory}' is not a valid directory. Cannot save asset.")
            return None
        output_path = os.path.join(self.output_directory, file_name)

        try:
            # Point-E doesn't directly use 'complexity' yet, but we have it for future use.
            # The description quality will primarily drive the output.
            point_cloud_data, figure = await self.point_e_handler.generate_point_cloud_from_text(
                prompt=description,
                output_path=output_path
            )
            
            if os.path.exists(output_path): # Check if file was actually saved
                new_asset = Asset(
                    agent_id=agent_id,
                    description=description,
                    file_path=output_path,
                    complexity=complexity
                )
                self.assets_created.append(new_asset)
                logger.info(f"AssetManager: Successfully created asset {new_asset.asset_id} for agent {agent_id} at {output_path}")
                return new_asset
            else:
                logger.warning(f"AssetManager: PointEHandler reported generation but file not found at {output_path}.")
                return None
                
        except Exception as e:
            logger.error(f"AssetManager: Error during asset generation for agent {agent_id} with prompt '{description}': {e}", exc_info=True)
            return None

if __name__ == '__main__':
    import asyncio

    async def main_test():
        print("Initializing AssetManager for testing...")
        manager = AssetManager(output_directory="test_generated_assets")
        
        test_agent_id = "agent_test_001"
        test_description = "a small blue cube"
        
        print(f"Attempting to create asset for {test_agent_id}: '{test_description}'")
        asset = await manager.create_asset_for_agent(test_agent_id, test_description)
        
        if asset:
            print(f"Asset creation successful: {asset}")
            print(f"Total assets created: {len(manager.assets_created)}")
            # Verify file exists
            if os.path.exists(asset.file_path):
                print(f"Verified asset file exists: {asset.file_path}")
            else:
                print(f"ERROR: Asset file NOT found: {asset.file_path}")
        else:
            print("Asset creation failed.")

        test_description_2 = "an ornate golden key"
        print(f"Attempting to create another asset for {test_agent_id}: '{test_description_2}'")
        asset_2 = await manager.create_asset_for_agent(test_agent_id, test_description_2, complexity="complex")
        if asset_2:
            print(f"Asset creation successful: {asset_2}")
            print(f"Total assets created: {len(manager.assets_created)}")
        else:
            print("Asset creation failed for the second asset.")


    asyncio.run(main_test()) 