"""
Blueprint: A simple blueprint class for code generation in blueprint mode.

This class maintains a simple dictionary that can be read, modified, and maintained
by different agents throughout the workflow. It serves as a reference for code generation
but does not maintain code generation state itself.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

class Blueprint:
    """
    Simple Blueprint class for maintaining simulation blueprint information across agents.
    
    This class provides a simple dictionary-based storage that can be modified by
    different agents during the workflow.
    """
    
    def __init__(self, task_description: str = ""):
        """
        Initialize the Blueprint with an empty dictionary.
        
        Args:
            task_description: The original task description (optional)
        """
        self.logger = logging.getLogger("SOCIA.Blueprint")
        
        # Simple dictionary to store blueprint data
        self.data = {}
        
        # Store task description if provided
        if task_description:
            self.data["created_at"] = datetime.now().isoformat()
            self.data["task_description"] = task_description
            # "Brief title for the simulation task"
            # self.data["title"] = ""
            # "<pick one: Economics | Sociology | Politics | Psychology | Organization | Demographics | Law | Communication>"
            # self.data["domain"] = ""
            # "agent_based | system_dynamics | network | etc."
            # self.data["simulation_type"] = ""
            # "TAMP"
            self.data["TAMP"] = {}
            self.data["TAMP"]["overview"] = {
      "phenomenon_and_objectives": "",
      "scale_and_granularity": "",
      "agent_archetypes": "",
      "interaction_topology": "",
    }
            self.data["TAMP"]["design_concepts"] = {
      "state_taxonomy": "",
      "decision_policy_family": "",
      "environment_and_signals": "",
      "actions_and_constraints": "",
    }
            self.data["TAMP"]["details"] = {
      "scenarios_and_interventions": "",
      "evidence_calibration_evaluation": "",
    }
        self.logger.info("Blueprint initialized with empty dictionary")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the blueprint dictionary.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value associated with the key, or default if not found
        """
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the blueprint dictionary.
        
        Args:
            key: The key to set
            value: The value to set
        """
        self.data[key] = value
        self.data["last_modified"] = datetime.now().isoformat()
        self.logger.debug(f"Set blueprint key: {key}")
    
    def update(self, other_dict: Dict[str, Any]) -> None:
        """
        Update the blueprint dictionary with another dictionary.
        
        Args:
            other_dict: Dictionary to merge into the blueprint
        """
        self.data.update(other_dict)
        self.data["last_modified"] = datetime.now().isoformat()
        self.logger.debug(f"Updated blueprint with {len(other_dict)} keys")
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get the complete blueprint data dictionary.
        
        Returns:
            Complete blueprint data dictionary
        """
        return self.data.copy()
    
    def to_json(self) -> str:
        """
        Convert blueprint to JSON string.
        
        Returns:
            JSON string representation of the blueprint
        """
        return json.dumps(self.data, indent=2, ensure_ascii=False)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save blueprint to a JSON file.
        
        Args:
            filepath: Path to save the blueprint file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
            self.logger.info(f"Blueprint saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving blueprint to file: {e}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load blueprint from a JSON file.
        
        Args:
            filepath: Path to load the blueprint file from
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.logger.info(f"Blueprint loaded from: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading blueprint from file: {e}")
    
    def clear(self) -> None:
        """Clear all data from the blueprint."""
        self.data.clear()
        self.logger.debug("Blueprint data cleared")
    
    def keys(self):
        """Get all keys in the blueprint."""
        return self.data.keys()
    
    def values(self):
        """Get all values in the blueprint."""
        return self.data.values()
    
    def items(self):
        """Get all key-value pairs in the blueprint."""
        return self.data.items()
    
    def __getitem__(self, key: str) -> Any:
        """Get item using bracket notation."""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using bracket notation."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in blueprint."""
        return key in self.data
    
    def __len__(self) -> int:
        """Get the number of items in the blueprint."""
        return len(self.data)
    
    def __str__(self) -> str:
        """String representation of the blueprint."""
        return f"Blueprint({len(self.data)} items)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the blueprint."""
        return f"Blueprint(data={self.data})"
