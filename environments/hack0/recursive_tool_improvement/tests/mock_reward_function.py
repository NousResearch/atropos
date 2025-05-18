"""
Mock implementation of the RewardFunction class from atroposlib.
"""

class RewardFunction:
    """
    Mock version of the Atropos RewardFunction class for testing purposes.
    """
    
    def __init__(self, weight: float = 1.0, name: str = None, **kwargs):
        """
        Initialize reward function with a weight and optional configuration.
        
        Args:
            weight: Importance factor when combining with other rewards
            name: Optional custom name for this reward function instance
            **kwargs: Additional configuration parameters specific to the reward function
        """
        self.weight = weight
        self._name = name
        self.config = kwargs
        self.wandb_logger = None
    
    @property
    def name(self) -> str:
        """Unique identifier for this reward function"""
        return self._name or self.__class__.__name__.lower()
    
    def compute(self, completions, **kwargs):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError
    
    def __call__(self, completions, **kwargs):
        """Wrapper that applies weight to the computed rewards"""
        try:
            rewards = self.compute(completions, **kwargs)
            # Apply weight
            weighted_rewards = [r * self.weight for r in rewards]
            return weighted_rewards
        except Exception as e:
            print(f"Error in reward function {self.name}: {e}")
            return [0.0] * len(completions)