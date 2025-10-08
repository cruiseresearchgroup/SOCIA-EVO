#!/usr/bin/env python3
"""
Test script for DataAnalysisOddAgent
Tests the data analysis odd functionality by running task understanding, data analysis, and model planning steps.
"""

import argparse
import logging
import os
import sys
import yaml
import json
from typing import Dict, Any, Optional
from orchestration.container import AgentContainer
from utils.llm_utils import load_api_key
from dependency_injector.wiring import Provide, inject

def setup_logging(output_path: Optional[str] = None, debug: bool = False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # If output path is provided, add a file handler
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Create log file path
            log_file_path = os.path.join(output_path, "test_data_analysis_odd.log")
            
            # Add file handler to handlers list
            handlers.append(logging.FileHandler(log_file_path))
            print(f"Logging to file: {log_file_path}")
        except Exception as e:
            print(f"Warning: Could not set up logging to file: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('SOCIA.DataAnalysisOddTest')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test script for SOCIA DataAnalysisOddAgent')
    parser.add_argument('--task', type=str, required=True, help='Description of the simulation task')
    parser.add_argument('--task-file', type=str, help='Path to task description JSON file')
    parser.add_argument('--output', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mode', type=str, default='full', choices=['lite', 'medium', 'full', 'blueprint', 'odd'], help='Workflow mode')
    parser.add_argument('--selfloop', type=int, default=3, help='Number of self-checking loop attempts for code generation')
    parser.add_argument('--persisted-data-analysis-file', type=str, help='Path to persisted data analysis file (task_spec.json) to skip data analysis phase')
    
    return parser.parse_args()

def setup_container(config_path: str) -> AgentContainer:
    """Set up and configure the dependency injection container."""
    container = AgentContainer()
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            container.config.from_dict(config)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        # Use minimal default configuration
        container.config.from_dict({
            "system": {"name": "SOCIA", "version": "0.1.0"},
            "agents": {
                "task_understanding": {"prompt_template": "templates/task_understanding_prompt.txt"},
                "data_analysis_odd": {"prompt_template": "templates/data_analysis_prompt.txt"},
                "code_generation_odd": {"prompt_template": "templates/code_generation_odd_prompt.txt"},
                "model_planning": {"prompt_template": "templates/model_planning_prompt.txt"}
            }
        })
    
    # Wire the container for dependency injection
    container.wire(modules=[
        sys.modules[__name__],
        "agents.task_understanding.agent",
        "agents.data_analysis_odd.agent", 
        "agents.code_generation_odd.agent",
        "agents.model_planning.agent",
        "agents.base_agent",
        "utils.llm_utils"
    ])
    
    return container

def load_task_data(task_file_path: str) -> Optional[Dict[str, Any]]:
    """Load task data from JSON file."""
    if not task_file_path or not os.path.exists(task_file_path):
        return None
    
    try:
        with open(task_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading task file {task_file_path}: {e}")
        return None

def extract_data_path_from_task_file(task_file_path: str, task_data: Dict[str, Any]) -> Optional[str]:
    """Extract data path from task file, resolving relative paths."""
    if not task_data or "data_folder" not in task_data:
        return None
    
    data_folder = task_data["data_folder"]
    
    # If data_folder is relative, make it relative to the task file directory
    if not os.path.isabs(data_folder):
        task_file_dir = os.path.dirname(os.path.abspath(task_file_path))
        project_root = os.getcwd()  # Assuming we're running from project root
        data_path = os.path.join(project_root, data_folder)
    else:
        data_path = data_folder
    
    return data_path

def save_artifact(output_path: str, name: str, data: Any):
    """Save an artifact to the output directory."""
    try:
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, f"{name}_iter_0.json")
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Saved {name} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving {name}: {e}")

def save_generated_code(output_path: str, generated_code: Dict[str, Any], iteration: int = 0):
    """Save generated code using dual persistence mechanism (JSON + Python file)."""
    logger = logging.getLogger()
    
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Save complete generated code data as JSON (includes metadata)
        json_file_path = os.path.join(output_path, f"generated_code_iter_{iteration}.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(generated_code, f, indent=2, default=str)
        logger.info(f"Saved generated code JSON to {json_file_path}")
        
        # 2. Save pure Python code as executable .py file
        if "code" in generated_code:
            code_content = generated_code["code"]
            code_file_path = os.path.join(output_path, f"simulation_code_iter_{iteration}.py")
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            logger.info(f"Saved Python code to {code_file_path}")
        else:
            logger.warning("No 'code' field found in generated_code data")
            
    except Exception as e:
        logger.error(f"Error saving generated code: {e}")
        raise

def load_persisted_data_analysis(file_path: str) -> Dict[str, Any]:
    """Load persisted data analysis from JSON file."""
    logger = logging.getLogger()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            task_spec = json.load(f)
        logger.info(f"Loaded persisted data analysis from: {file_path}")
        return task_spec
    except Exception as e:
        logger.error(f"Error loading persisted data analysis file {file_path}: {e}")
        raise

@inject
def run_data_analysis_test(
    args,
    logger,
    agent_container: AgentContainer = Provide[AgentContainer]
):
    """Run the data analysis test with data analysis only."""
    
    logger.info(f"Starting DataAnalysisOddAgent test in {args.mode.upper()} mode")
    
    try:
        # Set up output path in container
        agent_container.output_path.override(args.output)
        
        # Get agent instances
        agents = {
            "data_analysis": agent_container.data_analysis_odd_agent(),
            "code_generation": agent_container.code_generation_odd_agent()
        }
        
        # Check if we should skip data analysis and use persisted file
        if hasattr(args, 'persisted_data_analysis_file') and getattr(args, 'persisted_data_analysis_file', None):
            # Skip data analysis phase and load from persisted file
            logger.info("=" * 50)
            logger.info("LOADING PERSISTED DATA ANALYSIS")
            logger.info("=" * 50)
            
            task_spec = load_persisted_data_analysis(args.persisted_data_analysis_file)
            
            # Extract data path from task_spec if available
            data_path = None
            if "data_folder" in task_spec:
                data_folder = task_spec["data_folder"]
                if os.path.isabs(data_folder):
                    data_path = data_folder
                else:
                    # Construct absolute path using project root
                    data_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
                logger.info(f"Data path from persisted file: {data_path}")
                
            logger.info("Persisted data analysis loaded successfully")
            
        else:
            # Regular data analysis workflow
            task_data = None
            data_path = None
            
            if args.task_file:
                task_data = load_task_data(args.task_file)
                if task_data:
                    data_path = extract_data_path_from_task_file(args.task_file, task_data)
                    logger.info(f"Loaded task data from: {args.task_file}")
                    logger.info(f"Data path resolved to: {data_path}")
                else:
                    logger.warning(f"Could not load task data from: {args.task_file}")
            
            # Data Analysis (generates task_spec)
            logger.info("=" * 50)
            logger.info("DATA ANALYSIS WITH TASK SPEC GENERATION")
            logger.info("=" * 50)
            
            if task_data:
                task_spec = agents["data_analysis"].process(
                    task_description=args.task,
                    task_data=task_data
                )
            else:
                task_spec = agents["data_analysis"].process(
                    task_description=args.task
                )
        
        logger.info("Data analysis and task spec generation completed successfully")
        save_artifact(args.output, "task_spec", task_spec)
        
        # Log key task spec information
        logger.info("Task Specification Summary:")
        logger.info(f"  - Description: {task_spec.get('description', 'N/A')[:100]}...")
        logger.info(f"  - Data folder: {task_spec.get('data_folder', 'N/A')}")
        logger.info(f"  - Data files: {list(task_spec.get('data_files', {}).keys())}")
        
        # Log data analysis results if present
        data_analysis_result = task_spec.get('data_analysis_result', {})
        if data_analysis_result:
            logger.info("Data Analysis Summary:")
            logger.info(f"  - File summaries: {len(data_analysis_result.get('file_summaries', []))} files analyzed")
            logger.info(f"  - Overall simulation design: {bool(data_analysis_result.get('overall_simulation_design', {}))}")
            logger.info(f"  - Agent archetypes: {bool(data_analysis_result.get('agent_archetypes', {}))}")
            logger.info(f"  - Interaction topology: {bool(data_analysis_result.get('interaction_topology', {}))}")
            logger.info(f"  - Calibratable parameters: {len(data_analysis_result.get('calibratable_parameters', []))}")
            
            # Print first few file summaries for inspection
            file_summaries = data_analysis_result.get('file_summaries', [])
            if file_summaries:
                logger.info("First file summary preview:")
                first_summary = file_summaries[0]
                if isinstance(first_summary, dict):
                    summary_text = first_summary.get('semantic_summary', str(first_summary))
                    logger.info(f"  File: {first_summary.get('file_name', 'Unknown')}")
                    logger.info(f"  Type: {first_summary.get('file_type', 'Unknown')}")
                    logger.info(f"  Summary: {summary_text[:200]}...")
                else:
                    logger.info(f"  {str(first_summary)[:200]}...")
        
        # Code Generation using CodeGenerationOddAgent
        logger.info("=" * 50)
        logger.info("CODE GENERATION")
        logger.info("=" * 50)
        
        generated_code = agents["code_generation"].process(
            task_spec=task_spec,
            data_analysis=None,  # Not used in odd mode
            model_plan=None,     # Not used in odd mode
            feedback=None,       # No feedback in first iteration
            data_path=data_path,
            previous_code=None,  # No previous code in first iteration
            historical_fix_log=None,  # No historical fix log in first iteration
            mode=args.mode,      # Use mode from command line arguments
            selfloop=args.selfloop,  # Use selfloop from command line arguments
            blueprint=None       # No blueprint for odd agent
        )
        
        logger.info("Code generation completed successfully")
        # Use dual persistence mechanism (JSON + Python file)
        save_generated_code(args.output, generated_code, iteration=0)
        
        # Log generated code information
        logger.info("Generated Code Summary:")
        if isinstance(generated_code, dict):
            code_content = generated_code.get('code', '')
            code_summary = generated_code.get('code_summary', 'No summary available')
            metadata = generated_code.get('metadata', {})
            
            logger.info(f"  - Code length: {len(code_content)} characters")
            logger.info(f"  - Code summary: {code_summary}")
            logger.info(f"  - Model type: {metadata.get('model_type', 'N/A')}")
            logger.info(f"  - Mode: {metadata.get('mode', 'N/A')}")
            
            # Count lines of code
            if code_content:
                lines = code_content.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                logger.info(f"  - Total lines: {len(lines)}")
                logger.info(f"  - Non-empty lines: {len(non_empty_lines)}")
        else:
            logger.info(f"  - Generated code type: {type(generated_code)}")
        
        # Create state to store results (similar to workflow_manager)
        state = {
            "task_spec": task_spec,
            "data_analysis_result": data_analysis_result,
            "generated_code": generated_code
        }
        
        # Final summary
        logger.info("=" * 50)
        logger.info("TEST COMPLETION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Data Analysis: Completed")
        logger.info(f"✅ Code Generation: Completed")
        logger.info(f"📁 Artifacts saved to: {args.output}")
        
        return {
            "status": "success",
            "state": state,
            "task_spec": task_spec,
            "data_analysis": data_analysis_result,
            "generated_code": generated_code,
            "output_path": args.output
        }
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "output_path": args.output
        }

def check_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = load_api_key("OPENAI_API_KEY")
    return api_key is not None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output, args.debug)
    
    # Check API key
    if not check_api_key():
        logger.error("OpenAI API key not found in keys.py")
        logger.info("Please set up your API key using: python main.py --setup-api-key")
        return 1
    
    # Set up the dependency injection container
    container = setup_container(args.config)
    
    # Run the test
    result = run_data_analysis_test(args, logger)
    
    if result["status"] == "success":
        logger.info("DataAnalysisOddAgent test completed successfully!")
        return 0
    else:
        logger.error("DataAnalysisOddAgent test failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 