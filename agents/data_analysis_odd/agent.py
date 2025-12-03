"""
DataAnalysisAgent: Analyzes input data to extract patterns and calibration parameters.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd
import yaml

from agents.base_agent import BaseAgent
from utils.data_loader import DataLoader

class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent leverages LLM capabilities to analyze data, understand patterns,
    and extract parameters that can be used to calibrate simulation models.
    
    This agent is responsible for:
    1. Loading data files (assuming perfect data quality)
    2. Identifying key distributions and patterns in the data
    3. Extracting parameters that can be used to configure and calibrate simulations
    4. Using LLM to provide insights about how the data should inform model design and calibration
    """
    
    def __init__(self, config: Any, output_path: Optional[str] = None):
        super().__init__(config)
        # Base output path for persisting processed data
        self.output_path = output_path or os.getcwd()
    
    def process(
        self,
        task_description: str,
        task_data: Optional[Dict[str, Any]] = None,
        mode: str = "persona",
        blueprint: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process the input data and extract insights, and return task_spec with data analysis results.
        
        Args:
            task_description: Natural language description of the simulation task
            task_data: Task data from JSON task description file (optional)
            mode: Processing mode ('full' or 'blueprint') 
            blueprint: Blueprint object from workflow manager (optional)
        
        Returns:
            Dictionary containing task_spec with embedded data_analysis_result
        """
        self.logger.info(f"Processing task description: {task_description}")
        
        # Build task_spec from task_data if provided
        if task_data:
            self.logger.info("Using provided task data from JSON file")
            
            # Extract task objective
            task_objective = task_data.get("task_objective", {})
            description = task_objective.get("description", task_description)
            simulation_focus = task_objective.get("simulation_focus", [])
            
            # Extract data information
            data_folder = task_data.get("data_folder", "")
            data_files = task_data.get("data_files", {})
            self.logger.info(f"Data folder from task file: {data_folder}")
            self.logger.info(f"Data files specified: {list(data_files.keys())}")
            
            # Extract evaluation metrics
            evaluation_metrics = task_data.get("evaluation_metrics", {})
            
            # Create task specification from JSON data
            task_spec = {
                "title": "Simulation Task",
                "description": description,
                "simulation_focus": simulation_focus,
                "data_folder": data_folder,
                "data_files": data_files,
                "evaluation_metrics": evaluation_metrics,
            }
            # In persona mode, also surface additional high-level fields
            # from the task file so they can participate in blueprint generation.
            if mode == "persona":
                task_spec["tests"] = task_data.get("tests", {}) or {}
                task_spec["persona_specification"] = task_data.get("persona_specification", {}) or {}
            
            # Get data path from task_spec
            data_path = data_folder
            if not os.path.isabs(data_path):
                # Make relative path absolute
                data_path = os.path.join(os.getcwd(), data_folder)
        else:
            # No task data provided, create minimal task_spec
            task_spec = {
                "title": "Simulation Task",
                "description": task_description,
                "simulation_focus": [],
                "data_folder": "",
                "data_files": {},
                "evaluation_metrics": {}
            }
            # Ensure persona mode still has the expected keys, even without a task file
            if mode == "persona":
                task_spec["tests"] = {}
                task_spec["persona_specification"] = {}
            data_path = ""
        
        # Check if data path exists (only if data_path is provided)
        if data_path and not os.path.isdir(data_path):
            error_msg = f"Data path invalid or missing: {data_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if data_path:
            self.logger.info(f"Successfully verified data path exists: {data_path}")
        
        # Capture task description for semantic summaries
        if mode == "persona":
            # In persona mode, use the rich task specification (including
            # persona/tests/etc.) as the "description" for the LLM, but exclude
            # raw data location fields that are not semantically meaningful.
            task_spec_for_prompt = {
                key: value
                for key, value in task_spec.items()
                if key not in ("data_folder", "data_files")
            }
            # _build_prompt will JSON-encode dict values, so we can pass the
            # structure directly here.
            task_description_for_analysis = task_spec_for_prompt
        else:
            task_description_for_analysis = task_spec.get("description", task_description)
        
        # Skip data analysis if no data path provided
        if not data_path:
            self.logger.info("No data path provided, returning task_spec with empty data analysis result")
            task_spec["data_analysis_result"] = {
                "data_summary": {},
                "simulation_parameters": {},
                "calibration_strategy": {},
                "file_summaries": []
            }
            return task_spec
        
        # Create data loader
        data_loader = DataLoader(data_path)
        
        # Identify available data files
        available_files = self._list_available_files(data_path)
        self.logger.info(f"Found {len(available_files)} files in data path")
        
        # Determine which files to analyze based on task specification
        files_to_analyze = self._select_files_to_analyze(available_files, task_spec)
        self.logger.info(f"Selected {len(files_to_analyze)} files for analysis")
        
        # Check if data_files is specified in task_spec
        if task_spec and "data_files" in task_spec:
            expected_files = set(task_spec["data_files"].keys())
            found_files = {os.path.basename(f["path"]) for f in available_files}
            self.logger.info(f"Expected files: {expected_files}")
            self.logger.info(f"Found files (basename): {found_files}")
            missing_files = expected_files - found_files
            
            if missing_files:
                error_msg = f"Expected data files missing: {missing_files}"
                self.logger.error(error_msg)
                # Stop processing immediately if required files are missing
                raise FileNotFoundError(error_msg)
            else:
                self.logger.info(f"All expected files found in data directory")
                
            # Store current data path for schema inference
            self._current_data_path = data_path
            
            # Extract schema information from task_spec if available
            self.schemas = self._extract_schemas_from_task_spec(task_spec)
        else:
            self.schemas = {}

        
        # Iterate through schemas to generate semantic summaries
        for file_name, schema in self.schemas.items():
            self.logger.info(f"Generating semantic summary for: {file_name}")
            
            try:
                # Generate semantic summary using schema data instead of raw data
                self._get_semantic_summary(file_name, schema, task_description_for_analysis)
                self.logger.info(f"Successfully generated semantic summary for: {file_name}")
                
            except Exception as e:
                self.logger.error(f"Error generating semantic summary for {file_name}: {e}")
                # Continue processing other files
                continue
        
        # # Build prompt for LLM analysis
        # metrics = task_spec.get("metrics", [])
        # metrics_description = json.dumps(metrics, indent=2) if metrics else "No metrics specified"
        #
        # # Create context about simulation calibration
        # calibration_context = self._create_calibration_context(task_spec, {})
        
        # Extract and format file_semantic_summary from schemas
        formatted_file_summaries = []
        for file_name, schema in self.schemas.items():
            file_type = schema.get("file_type", "unknown")
            file_semantic_summary = schema.get("file_semantic_summary", "No semantic summary available")
            
            formatted_summary = {
                "file_name": file_name,
                "file_type": file_type,
                "semantic_summary": file_semantic_summary
            }
            formatted_file_summaries.append(formatted_summary)
        
        # --------------------------------------------------
        # Build data_analysis_result
        # --------------------------------------------------
        if mode == "persona":
            # Persona mode: do NOT call the LLM for blueprint. Instead,
            # use the original task file content (task_data) directly as the
            # blueprint, excluding raw data-location fields. Downstream code
            # generation and feedback will operate on this persona/test spec.
            if task_data:
                self.logger.info(
                    "Persona mode: building data_analysis_result from task_data "
                    "excluding 'data_folder' and 'data_files'"
                )
                data_analysis_result = {
                    key: value
                    for key, value in task_data.items()
                    if key not in ("data_folder", "data_files")
                }
            else:
                self.logger.warning(
                    "Persona mode enabled but no task_data provided; using empty data_analysis_result"
                )
                data_analysis_result = {}
        else:
            # Non-persona modes: call the LLM to analyse data and map the response
            # into the standard 10-key schema.
            analysis_prompt = self._build_analysis_prompt(
                task_description=task_description_for_analysis,
                data_schemas=self.schemas
            )
            self.logger.info("Calling LLM to analyze data and provide calibration insights")
            llm_response = self._call_llm(analysis_prompt)
            analysis_results = self._parse_llm_analysis(llm_response)
            
        data_analysis_result = {
            "overall_simulation_design": analysis_results.get("overall_simulation_design", {}),
            "scale_granularity": analysis_results.get("scale_granularity", {}),
            "agent_archetypes": analysis_results.get("agent_archetypes", {}),
            "interaction_topology": analysis_results.get("interaction_topology", {}),
            "information_propagation": analysis_results.get("information_propagation", {}),
            "exogenous_signals": analysis_results.get("exogenous_signals", []),
            "action_decision_policy": analysis_results.get("action_decision_policy", {}),
            "holdout_plan": analysis_results.get("holdout_plan", {}),
            "simulation_evaluation": analysis_results.get("simulation_evaluation", {}),
            "calibratable_parameters": analysis_results.get("calibratable_parameters", []),
        }
        
        # Add data analysis result and file summaries to task_spec
        task_spec["data_analysis_result"] = data_analysis_result
        task_spec["file_summaries"] = formatted_file_summaries
        
        # Add generated schemas to task_spec if available
        if hasattr(self, 'schemas') and self.schemas:
            task_spec["schemas"] = self.schemas
            self.logger.info(f"Added {len(self.schemas)} file schemas to task_spec")
        
        self.logger.info("Data analysis completed")
        return task_spec
    
    def _extract_schemas_from_task_spec(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract schema information from task specification if available.
        
        Args:
            task_spec: Task specification which may contain schema information
            
        Returns:
            Dictionary mapping file names to their schemas
        """
        schemas = {}
        
        # Check if schemas are directly provided in task_spec
        if "schemas" in task_spec:
            return task_spec["schemas"]
        
        # Try to infer schemas from data_files descriptions
        if "data_files" in task_spec:
            for file_name, description in task_spec["data_files"].items():
                # If description is a string, try to infer schema from it using LLM
                if isinstance(description, str):
                    schema = self._infer_schema_from_description(file_name, description)
                    if schema:
                        schemas[file_name] = schema
                # If description is already a structured object with schema
                elif isinstance(description, dict) and "schema" in description:
                    schemas[file_name] = description["schema"]
                    
        return schemas
    
    def _infer_schema_from_description(self, file_name: str, description: str) -> Dict[str, Any]:
        """
        Infer schema by actually examining the data file.
        
        Args:
            file_name: Name of the file
            description: Description of the file content
            
        Returns:
            Actual data schema based on file content
        """
        self.logger.info(f"Inferring schema for {file_name} by examining actual data")
        
        # Get the data path from current processing context
        data_path = getattr(self, '_current_data_path', None)
        if not data_path:
            self.logger.warning(f"No data path available for schema inference of {file_name}")
            return {"type": "object", "additionalProperties": True}
        
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found for schema inference: {file_path}")
            return {"type": "object", "additionalProperties": True}
        
        # Determine file type
        file_extension = os.path.splitext(file_name)[1].lower()
        
        try:
            if file_extension == '.csv':
                return self._infer_csv_schema(file_path, file_name)
            elif file_extension == '.json':
                return self._infer_json_schema(file_path, file_name)
            elif file_extension in {'.yaml', '.yml'}:
                return self._infer_yaml_schema(file_path, file_name)
            else:
                self.logger.info(f"Unsupported file type for schema inference: {file_extension}")
                return {"type": "object", "additionalProperties": True}
                
        except Exception as e:
            self.logger.error(f"Error inferring schema for {file_name}: {e}")
            return {"type": "object", "additionalProperties": True}
    
    def _infer_csv_schema(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Infer schema from CSV file by examining its structure and content.
        
        Args:
            file_path: Path to the CSV file
            file_name: Name of the file
            
        Returns:
            Schema dictionary containing CSV analysis
        """
        self.logger.info(f"Analyzing CSV structure for {file_name}")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Get basic info - capture df.info() output
        import io
        import sys
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        # Get row count
        row_count = len(df)
        
        # Get first 10 rows as sample data
        sample_data = df.head(10).to_dict('records')
        
        schema = {
            "file_type": "csv",
            "row_count": int(row_count),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "data_info": info_str,
            "sample_data": sample_data,
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        }
        
        self.logger.info(f"CSV schema generated for {file_name}: {row_count} rows, {len(df.columns)} columns")
        return schema
    
    def _infer_json_schema(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Infer schema from JSON file by examining its structure and content.
        
        Args:
            file_path: Path to the JSON file
            file_name: Name of the file
            
        Returns:
            Schema dictionary containing JSON analysis
        """
        self.logger.info(f"Analyzing JSON structure for {file_name}")
        
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get length and sample data based on data structure
        if isinstance(data, list):
            data_length = len(data)
            sample_data = data[:10] if len(data) >= 10 else data
            data_type = "array"
        elif isinstance(data, dict):
            data_length = len(data)
            # Get first 10 items from dictionary
            sample_data = dict(list(data.items())[:10])
            data_type = "object"
        else:
            data_length = 1
            sample_data = data
            data_type = type(data).__name__
        
        schema = {
            "file_type": "json",
            "data_type": data_type,
            "length": data_length,
            "sample_data": sample_data
        }
        
        # Add additional structure info for objects and arrays
        if isinstance(data, dict):
            if data:
                # Analyze structure of first item
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                schema["value_structure"] = {
                    "type": type(first_value).__name__,
                    "sample_value": first_value
                }
        elif isinstance(data, list) and data:
            # Analyze structure of first item in array
            first_item = data[0]
            schema["item_structure"] = {
                "type": type(first_item).__name__,
                "sample_item": first_item
            }
        
        self.logger.info(f"JSON schema generated for {file_name}: {data_type} with {data_length} items")
        return schema
    
    def _infer_yaml_schema(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Infer schema from YAML file by examining its full content.
        """
        self.logger.info(f"Analyzing YAML structure for {file_name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file {file_name}: {e}")
            return {
                "file_type": "yaml",
                "data_type": "unparseable",
                "raw_text": raw_text,
                "sample_data": raw_text
            }
        
        if data is None:
            return {
                "file_type": "yaml",
                "data_type": "empty",
                "raw_text": raw_text,
                "sample_data": ""
            }
        
        schema: Dict[str, Any] = {
            "file_type": "yaml",
            "raw_text": raw_text
        }
        
        if isinstance(data, dict):
            schema["data_type"] = "object"
            schema["length"] = len(data)
            schema["sample_data"] = dict(list(data.items())[:10])
            schema["keys"] = list(data.keys())
        elif isinstance(data, list):
            schema["data_type"] = "array"
            schema["length"] = len(data)
            schema["sample_data"] = data[:10]
        else:
            schema["data_type"] = type(data).__name__
            schema["sample_data"] = data
        
        self.logger.info(f"YAML schema generated for {file_name}: {schema.get('data_type')} with length {schema.get('length', 'n/a')}")
        return schema
    
    def _list_available_files(self, data_path: str) -> List[Dict[str, str]]:
        """List available files in the data directory."""
        result = []
        
        self.logger.info(f"Scanning directory for files: {data_path}")
        for root, dirs, files in os.walk(data_path):
            self.logger.info(f"Examining directory: {root}, contains {len(files)} files")
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_path)
                
                self.logger.debug(f"Checking file: {full_path}")
                
                if file.endswith('.csv'):
                    result.append({"path": rel_path, "type": "csv"})
                elif file.endswith('.json'):
                    if self._is_geojson(full_path):
                        result.append({"path": rel_path, "type": "geojson"})
                    else:
                        result.append({"path": rel_path, "type": "json"})
                elif file.endswith('.geojson'):
                    result.append({"path": rel_path, "type": "geojson"})
                elif file.lower().endswith('.pkl'):
                    # Pickle files (e.g., network data) - ensure case insensitive matching
                    self.logger.info(f"Found pickle file: {full_path}")
                    result.append({"path": rel_path, "type": "pkl"})
                elif file.endswith('.py'):
                    self.logger.info(f"Found Python file: {full_path}")
                    result.append({"path": rel_path, "type": "py"})
                elif file.endswith('.yaml') or file.endswith('.yml'):
                    result.append({"path": rel_path, "type": "yaml"})
        
        self.logger.info(f"Total files found: {len(result)} in path {data_path}")
        for f in result:
            self.logger.info(f"Found file in result: {f['path']} (type: {f['type']})")
        
        return result
    
    def _is_geojson(self, file_path: str) -> bool:
        """Check if a JSON file is a GeoJSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return "type" in data and "features" in data
        except:
            return False
    
    def _select_files_to_analyze(
        self,
        available_files: List[Dict[str, str]],
        task_spec: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Select which files to analyze based on task specification."""
        # Only analyze files specified in task_spec["data_files"] if provided.
        if task_spec and "data_files" in task_spec:
            expected_files = set(task_spec["data_files"].keys())
            selected = [f for f in available_files if os.path.basename(f["path"]) in expected_files]
            # Log files that are skipped because they are not in the spec
            skipped = [f["path"] for f in available_files if os.path.basename(f["path"]) not in expected_files]
            for skip in skipped:
                self.logger.info(f"Skipping file not in task_spec data_files: {skip}")
            return selected
        # Default: return all files if no data_files key in task_spec
        return available_files
    
    def _create_file_summary(
        self, 
        file_name: str, 
        data: pd.DataFrame,
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a concise summary of the file for inclusion in the LLM prompt.
        """
        column_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        column_descriptions = file_info.get("column_descriptions", {})
        transformations = file_info.get("transformations", {})
        
        # Create statistical summaries for numeric columns
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    stats[col] = {
                        "min": float(data[col].min()),
                        "max": float(data[col].max()),
                        "mean": float(data[col].mean()),
                        "median": float(data[col].median()),
                        "std": float(data[col].std())
                    }
                except:
                    # Skip if we can't compute statistics
                    pass
        
        # For boolean columns, calculate proportion of True values
        bool_props = {}
        for col in data.columns:
            if pd.api.types.is_bool_dtype(data[col]):
                try:
                    bool_props[col] = float(data[col].mean())  # Proportion of True values
                except:
                    pass
        
        # Combine all information
        summary = {
            "file_name": file_name,
            "purpose": file_info.get("purpose", "Unknown purpose"),
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "column_types": column_types,
            "column_descriptions": column_descriptions,
            "transformations": transformations,
            "statistics": stats,
            "boolean_proportions": bool_props,
            "key_insights": file_info.get("key_insights", [])
        }
        
        return json.dumps(summary, indent=2)
    
    def _get_json_structure(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a simplified representation of the JSON structure.
        """
        if isinstance(json_data, dict):
            return {
                "type": "object",
                "keys": list(json_data.keys()),
                "sample_values": {
                    k: type(v).__name__ for k, v in list(json_data.items())[:10]
                }
            }
        elif isinstance(json_data, list) and json_data:
            return {
                "type": "array",
                "length": len(json_data),
                "sample_item_type": type(json_data[0]).__name__
            }
        else:
            return {"type": type(json_data).__name__}
    
    def _create_json_summary(
        self, 
        file_name: str, 
        json_data: Dict[str, Any],
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a summary of a JSON file.
        """
        structure = file_info.get("structure", {})
        
        # Create a simplified summary
        summary = {
            "file_name": file_name,
            "type": "json",
            "structure": structure
        }
        
        return json.dumps(summary, indent=2)
    
    def _get_pickle_info(self, data: Any, file_name: str) -> Dict[str, Any]:
        """
        Get information about pickle data.
        """
        data_type = type(data).__name__
        
        if hasattr(data, "shape"):  # For numpy arrays
            info = {
                "type": "numpy_array",
                "shape": str(data.shape),
                "dtype": str(data.dtype)
            }
        elif hasattr(data, "nodes"):  # For networkx graphs
            info = {
                "type": "graph",
                "num_nodes": len(data.nodes),
                "num_edges": len(data.edges)
            }
        elif isinstance(data, dict):
            info = {
                "type": "dictionary",
                "num_keys": len(data),
                "key_types": list(set(type(k).__name__ for k in data.keys()))
            }
        elif isinstance(data, list):
            info = {
                "type": "list",
                "length": len(data)
            }
        else:
            info = {
                "type": data_type
            }
        
        return info
    
    def _create_pickle_summary(
        self, 
        file_name: str, 
        data: Any,
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a summary of a pickle file.
        """
        # Create a simplified summary
        summary = {
            "file_name": file_name,
            "type": "pickle",
            "info": file_info
        }
        
        return json.dumps(summary, indent=2)
    
    def _create_calibration_context(
        self,
        task_spec: Dict[str, Any],
        file_info_dict: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Create context about simulation calibration for the LLM.
        """
        # Extract relevant information from task specification
        task_description = task_spec.get("description", "")
        metrics = task_spec.get("metrics", [])
        
        # Extract file purposes
        file_purposes = {
            file_name: info.get("purpose", "Unknown purpose")
            for file_name, info in file_info_dict.items()
        }
        
        # Create context
        context = {
            "task_description": task_description,
            "metrics": metrics,
            "file_purposes": file_purposes
        }
        
        return json.dumps(context, indent=2)
    
    def _build_analysis_prompt(
        self,
        task_description: str,
        data_schemas: Dict[str, Any]
    ) -> str:
        """
        Build a comprehensive prompt for the LLM to analyze the data and provide insights.
        """
        # Build file summaries from schemas
        file_summaries = []
        for file_name, schema in data_schemas.items():
            file_type = schema.get("file_type", "unknown")
            file_semantic_summary = schema.get("file_semantic_summary", "No semantic summary available")
            file_examples = schema.get("sample_data", None)
            file_summary = f"File: {file_name} (type: {file_type})\n\nExample data of this file: \n{file_examples}\n\nSemantic summary of this file: \n{file_semantic_summary}"
            file_summaries.append(file_summary)
        
        file_summaries_text = "\n\n".join(file_summaries)
        
        # Build prompt from configured template; fallback to odd prompt file if missing
        if not self.prompt_template:
            try:
                template_path = os.path.join("templates", "data_analysis_odd_prompt.txt")
                self.logger.warning(f"No prompt template configured; falling back to {template_path}")
                with open(template_path, 'r') as f:
                    self.prompt_template = f.read()
            except Exception as e:
                self.logger.error(f"Error loading fallback prompt template: {e}")
                self.prompt_template = ""
        
        prompt = self._build_prompt(
            task_description=task_description,
            file_summaries_text=file_summaries_text
        )
        return prompt
    
    def _parse_llm_analysis(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response into a structured format.
        """
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Transform the new JSON format to match the expected data_analysis_result structure
                transformed_analysis = {
                    "overall_simulation_design": analysis.get("overall_simulation_design", {}),
                    "scale_granularity": analysis.get("scale_granularity", {}),
                    "agent_archetypes": analysis.get("agent_archetypes", {}),
                    "interaction_topology": analysis.get("interaction_topology", {}),
                    "information_propagation": analysis.get("information_propagation", {}),
                    "exogenous_signals": analysis.get("exogenous_signals", []),
                    "action_decision_policy": analysis.get("action_decision_policy", {}),
                    "holdout_plan": analysis.get("holdout_plan", {}),
                    "simulation_evaluation": analysis.get("simulation_evaluation", {}),
                    "calibratable_parameters": analysis.get("calibratable_parameters", [])
                }
                
                return transformed_analysis
            else:
                self.logger.warning("Could not extract JSON from analysis response")
                return {
                    "overall_simulation_design": {},
                    "scale_granularity": {},
                    "agent_archetypes": {},
                    "interaction_topology": {},
                    "information_propagation": {},
                    "exogenous_signals": [],
                    "action_decision_policy": {},
                    "holdout_plan": {},
                    "simulation_evaluation": {},
                    "calibratable_parameters": []
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return {
                "overall_simulation_design": {},
                "scale_granularity": {},
                "agent_archetypes": {},
                "interaction_topology": {},
                "information_propagation": {},
                "exogenous_signals": [],
                "action_decision_policy": {},
                "holdout_plan": {},
                "simulation_evaluation": {},
                "calibratable_parameters": []
            }
    
    def _convert_df_to_serializable(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert a pandas DataFrame to a serializable format.
        """
        # For datetime columns, convert to strings
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        # Convert to dictionary
        return df.to_dict(orient='list')
    
    def _get_semantic_summary(self, file_name: str, schema: Dict[str, Any], task_description: str) -> None:
        """
        Use LLM to generate a concise semantic metadata summary based on schema information.
        The summary is stored directly in the schema under 'file_semantic_summary' key.
        """
        # Extract sample data or raw text from schema
        if schema.get("file_type") == "yaml" and schema.get("raw_text"):
            sample_str = schema["raw_text"]
        else:
            sample_data = schema.get("sample_data", [])
            sample_str = json.dumps(sample_data, indent=2) if sample_data else "No sample data available"
        
        prompt = (
            f"Task Description: {task_description}\n\n"
            f"File: {file_name}\n"
            f"Data sample:\n{sample_str}\n\n"
            "Please provide a concise semantic metadata summary of this file in the context of the task, addressing:\n"
            "- Overall data structure and type\n"
            "- Relationships or nested elements\n"
            "- How this data should inform simulation entities or interactions\n"
        )
        self.logger.info(f"Generating semantic summary for {file_name}")
        llm_response = self._call_llm(prompt)
        
        # Store the semantic summary directly in the schema
        schema["file_semantic_summary"] = llm_response.strip()
    
