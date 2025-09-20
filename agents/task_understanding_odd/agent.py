"""
TaskUnderstandingAgent: Analyzes user requirements and converts them into structured simulation specifications.
"""

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from core.blueprint import Blueprint

class TaskUnderstandingAgent(BaseAgent):
    """
    Task Understanding Agent analyzes user requirements and converts them into structured
    simulation specifications that other agents can use.
    
    This agent is responsible for:
    1. Extracting key simulation requirements from natural language descriptions
    2. Identifying required entities, behaviors, and interactions
    3. Determining appropriate metrics and success criteria
    4. Structuring all this information into a consistent format for downstream agents
    """
    
    def process(self, task_description: str, task_data: Optional[Dict[str, Any]] = None, mode: str = "full", blueprint: Optional[Any] = None, task_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the task description and extract structured simulation requirements.
        
        Args:
            task_description: Natural language description of the simulation task
            task_data: Task data from JSON task description file (optional)
            mode: Processing mode ('full' or 'blueprint')
            blueprint: Blueprint object from workflow manager (optional)
            task_spec: Task specification with embedded data analysis (optional, defaults to empty dict)
        
        Returns:
            Dictionary containing structured simulation requirements
        """
        self.logger.info(f"Processing task description in {mode} mode")
        
        # Initialize task_spec to empty dict if not provided
        if task_spec is None:
            task_spec = {}
            self.logger.info("No task_spec provided, initializing empty dict")
        else:
            self.logger.info("Using provided task_spec as input")
        
        # Special handling for blueprint mode with dual-layer prompt design
        if mode == "blueprint":
            return self._process_blueprint_mode(task_description, task_data, blueprint)
        
        # Standard processing for full mode
        # Process and enhance the provided task_spec
        self.logger.info("Processing and enhancing provided task_spec")
        
        # Build enhancement prompt based on available information
        if task_data:
            self.logger.info("Using task data to enhance task understanding")
            enhanced_prompt = self._build_prompt(
                task_description=task_description,
                task_data=task_data
            )
        else:
            self.logger.info("Using task description only for enhancement")
            enhanced_prompt = self._build_prompt(task_description=task_description)
        
        # Call LLM to enhance the task specification
        try:
            llm_response = self._call_llm(enhanced_prompt)
            enhanced_spec = self._parse_llm_response(llm_response)
            
            # Merge enhanced specification with existing task_spec
            if isinstance(enhanced_spec, dict):
                self.logger.info("Merging LLM enhancements with existing task_spec")
                for key, value in enhanced_spec.items():
                    if key not in task_spec or not task_spec[key]:
                        task_spec[key] = value
                    elif isinstance(task_spec[key], dict) and isinstance(value, dict):
                        # Deep merge for nested dictionaries
                        task_spec[key].update(value)
            else:
                self.logger.warning("LLM response was not parsed as dict, adding as enhancement note")
                task_spec["enhancement_note"] = str(enhanced_spec) if enhanced_spec else "No enhancement available"
                
        except Exception as e:
            self.logger.error(f"Failed to enhance task_spec with LLM: {e}")
            task_spec["enhancement_error"] = str(e)
        
        # Final validation: ensure task_spec is JSON-serializable
        try:
            import json
            json.dumps(task_spec)
            self.logger.info("Task understanding completed - enhanced task_spec verified as JSON-serializable")
        except (TypeError, ValueError) as e:
            self.logger.error(f"Enhanced task_spec is not JSON-serializable: {e}")
            task_spec["serialization_error"] = str(e)
        
        return task_spec
    
    def _process_blueprint_mode(self, task_description: str, task_data: Optional[Dict[str, Any]] = None, blueprint: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process task using blueprint mode with TAMP axis iterative processing.
        
        Args:
            task_description: Natural language description of the simulation task
            task_data: Task data from JSON task description file (optional)
            blueprint: Blueprint object from workflow manager (optional)
            
        Returns:
            Dictionary containing structured simulation requirements with TAMP structure
        """
        self.logger.info("Processing in blueprint mode with TAMP axis iteration")
        
        # Determine the original research topic
        if task_data and "task_objective" in task_data:
            # Extract from task_objective field if available
            task_objective = task_data["task_objective"]
            if isinstance(task_objective, dict):
                result = []
                for key, value in task_objective.items():
                    if isinstance(value, list):
                        result.append(f"{key}: " + "\n".join(str(v) for v in value))
                    else:
                        result.append(f"{key}: {str(value)}")
                original_topic = "\n".join(result)
            elif isinstance(task_objective, str):
                original_topic = task_objective
            else:
                original_topic = task_description
        else:
            # Use task_description directly
            original_topic = task_description
        
        # Load blueprint-specific prompt template
        blueprint_template_path = "templates/task_understanding_blueprint_prompt.txt"
        try:
            with open(blueprint_template_path, 'r') as f:
                prompt_template = f.read()
            self.logger.info("Loaded blueprint-specific prompt template")
        except Exception as e:
            self.logger.error(f"Error loading blueprint template: {e}, falling back to standard template")
            prompt_template = self.prompt_template
        
        # Use provided blueprint or create a new one for progressive filling
        if blueprint is None:
            blueprint = Blueprint(original_topic)
        else:
            # Initialize the provided blueprint with original topic
            blueprint.set("task_description", original_topic)
            blueprint.set("created_at", blueprint.data.get("created_at", ""))
        
        # Define TAMP axis mapping to blueprint structure
        axis_mapping = {
            1: ("overview", "phenomenon_and_objectives"),
            2: ("overview", "scale_and_granularity"), 
            3: ("overview", "agent_archetypes"),
            4: ("design_concepts", "state_taxonomy"),
            5: ("overview", "interaction_topology"),
            6: ("design_concepts", "decision_policy_family"),
            7: ("design_concepts", "environment_and_signals"),
            8: ("design_concepts", "actions_and_constraints"),
            9: ("details", "scenarios_and_interventions"),
            10: ("details", "evidence_calibration_evaluation")
        }
        
        # Process each TAMP axis iteratively
        for axis_num in range(1, 11):
            try:
                self.logger.info(f"Processing TAMP axis {axis_num}")
                
                # Load the specific axis content
                axis_content = self._load_tamp_axis(axis_num)
                
                # Build prompt with current blueprint state
                prompt = prompt_template.format(
                    task_description=original_topic,
                    tamp_blueprint=self._format_tamp_blueprint(blueprint.get("TAMP")),
                    design_axis=axis_content
                )
                
                # Call LLM for this specific axis
                llm_response = self._call_llm(prompt)
                
                # Clean and extract the response (should be a single string)
                axis_response = self._clean_axis_response(llm_response)
                
                # Fill the response into the appropriate blueprint location
                section, field = axis_mapping[axis_num]
                blueprint.data["TAMP"][section][field] = axis_response
                
                self.logger.info(f"Axis {axis_num} response filled into {section}.{field}")
                
            except Exception as e:
                self.logger.error(f"Error processing TAMP axis {axis_num}: {e}")
                # Continue with next axis even if one fails
                section, field = axis_mapping[axis_num]
                blueprint.data["TAMP"][section][field] = f"Error processing axis {axis_num}: {str(e)}"
        
        # Add final metadata
        blueprint.set("processing_mode", "blueprint")
        blueprint.set("tamp_protocol", True)
        blueprint.set("original_research_topic", original_topic)
        
        # Add task data information if available
        if task_data:
            if "data_folder" in task_data:
                blueprint.set("data_folder", task_data["data_folder"])
            if "data_files" in task_data:
                blueprint.set("data_files", task_data["data_files"])
            if "evaluation_metrics" in task_data:
                blueprint.set("evaluation_metrics", task_data["evaluation_metrics"])
        
        self.logger.info("Blueprint mode task understanding completed - TAMP structure filled")
        return blueprint.get_data()
    
    def _load_tamp_axis(self, axis_num: int) -> str:
        """
        Load TAMP axis content from template files.
        
        Args:
            axis_num: Axis number (1-10)
            
        Returns:
            Content of the specified TAMP axis
        """
        axis_file_path = f"templates/tamp_axes/tamp_axis_{axis_num}.txt"
        try:
            with open(axis_file_path, 'r', encoding='utf-8') as f:
                axis_content = f.read()
            self.logger.info(f"Loaded TAMP axis {axis_num} from {axis_file_path}")
            return axis_content
        except Exception as e:
            self.logger.error(f"Error loading TAMP axis {axis_num}: {e}")
            return f"Error loading axis {axis_num}: {str(e)}"
    
    def _format_tamp_blueprint(self, tamp_data: Dict[str, Any]) -> str:
        """
        Format TAMP blueprint data for prompt inclusion.
        
        Args:
            tamp_data: Current TAMP data structure from Blueprint object
            
        Returns:
            Formatted string representation of TAMP blueprint
        """
        import json
        try:
            return json.dumps(tamp_data, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error formatting TAMP blueprint: {e}")
            return str(tamp_data)
    
    def _clean_axis_response(self, llm_response: str) -> str:
        """
        Clean and extract the axis response from LLM output.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Cleaned response string suitable for blueprint filling
        """
        # Remove common LLM response artifacts
        response = llm_response.strip()
        
        # Remove markdown formatting if present
        if response.startswith('```') and response.endswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        
        # Remove quotes if the entire response is quoted
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]
        
        # Truncate according to provider-effective max tokens by word count (space split)
        # Use cached effective max tokens from agent initialization
        effective_max_tokens = getattr(self, 'effective_max_tokens', 2000)
        words = response.split()
        if len(words) > effective_max_tokens:
            self.logger.warning(
                f"Axis response has {len(words)} words > token limit {effective_max_tokens}, truncating by words"
            )
            response = " ".join(words[:effective_max_tokens]) + " ..."
        
        return response.strip()