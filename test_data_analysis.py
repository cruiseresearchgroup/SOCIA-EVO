#!/usr/bin/env python3
"""
Test script for DataAnalysisOddAgent
Tests the data analysis odd functionality by running task understanding, data analysis, and model planning steps.
"""

import argparse
import logging
import os
import re
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
    parser.add_argument('--mode', type=str, default='persona', choices=['lite', 'medium', 'persona', 'blueprint', 'odd'], help='Workflow mode')
    parser.add_argument('--selfloop', type=int, default=3, help='Number of self-checking loop attempts for code generation')
    parser.add_argument('--persisted-data-analysis-file', type=str, help='Path to persisted data analysis file (task_spec.json) to skip data analysis phase')
    parser.add_argument('--persisted-code-file', type=str, help='Path to persisted code file (simulation_code_iter_N.py) to skip data analysis and initial code generation')
    parser.add_argument('--auto', action='store_true', default=False, help='Enable automatic mode; when False, user will be prompted to input feedback manually in each iteration')
    parser.add_argument('--iterations', type=int, default=3, help='Maximum number of iterations')
    
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
                "data_analysis_odd": {"prompt_template": "templates/data_analysis_odd_prompt.txt"},
                "code_generation_odd": {"prompt_template": "templates/code_generation_odd_prompt.txt"},
                "model_planning": {"prompt_template": "templates/model_planning_prompt.txt"},
                "code_verification": {"prompt_template": "templates/code_verification_prompt.txt"},
                "simulation_execution": {"prompt_template": "templates/simulation_execution_prompt.txt"},
                "result_evaluation": {"prompt_template": "templates/result_evaluation_prompt.txt"},
                "feedback_generation": {"prompt_template": "templates/feedback_generation_prompt.txt"},
                "feedback_generation_odd": {"prompt_template": "templates/feedback_generation_prompt.txt"},
                "iteration_control": {"prompt_template": "templates/iteration_control_prompt.txt"}
            }
        })
    
    # Wire the container for dependency injection
    container.wire(modules=[
        sys.modules[__name__],
        "agents.task_understanding.agent",
        "agents.data_analysis_odd.agent", 
        "agents.code_generation_odd.agent",
        "agents.model_planning.agent",
        "agents.code_verification.agent",
        "agents.simulation_execution.agent",
        "agents.result_evaluation.agent",
        "agents.feedback_generation.agent",
        "agents.feedback_generation_odd.agent",
        "agents.iteration_control.agent",
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

def save_artifact(output_path: str, name: str, data: Any, iteration: Optional[int] = None):
    """Save an artifact to the output directory.
    
    Args:
        output_path: Directory to save the artifact
        name: Base name for the artifact (without iteration suffix)
        data: Data to save
        iteration: Optional iteration number. If None and name doesn't contain _iter_, defaults to 0.
                   If name already contains _iter_X, uses that iteration number.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # Check if name already contains _iter_ pattern
        iter_pattern = r'_iter_(\d+)$'
        match = re.search(iter_pattern, name)
        
        if match:
            # Name already contains iteration number, use it directly
            file_path = os.path.join(output_path, f"{name}.json")
        elif iteration is not None:
            # Use provided iteration number
            file_path = os.path.join(output_path, f"{name}_iter_{iteration}.json")
        else:
            # Default to iter_0
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

def load_persisted_code(file_path: str) -> str:
    """Load persisted code from Python file."""
    logger = logging.getLogger()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        logger.info(f"Loaded persisted code from: {file_path}")
        return code
    except Exception as e:
        logger.error(f"Error loading persisted code file {file_path}: {e}")
        raise

def regenerate_blueprint_from_task_file(
    data_analysis_agent,
    task_spec: Dict[str, Any],
    task_file_path: str,
    logger
) -> Dict[str, Any]:
    """
    Rebuild the data analysis blueprint by feeding the full task file content
    into the personality-specific prompt template.
    """
    if not task_file_path or not os.path.exists(task_file_path):
        raise FileNotFoundError(f"Task file not found: {task_file_path}")
    
    template_path = data_analysis_agent.config.get(
        "prompt_template",
        "templates/data_analysis_odd_prompt_personality.txt"
    )
    if not os.path.isabs(template_path):
        template_path = os.path.abspath(template_path)
    
    logger.info("Full mode: Regenerating blueprint using personality template")
    logger.info(f"  → Template: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    with open(task_file_path, 'r', encoding='utf-8') as f:
        task_file_content = f.read()
    
    file_summaries = task_spec.get("file_summaries", [])
    if file_summaries:
        file_summaries_text = json.dumps(file_summaries, indent=2, default=str)
    else:
        file_summaries_text = "No file summaries available"
    
    prompt = prompt_template.replace("{task_file}", task_file_content)
    prompt = prompt.replace("{file_summaries_text}", file_summaries_text)
    
    llm_response = data_analysis_agent._call_llm(prompt, reasoning={"effort": "medium"})
    logger.info("Full mode: Received blueprint response from LLM")
    
    analysis_results = data_analysis_agent._parse_llm_analysis(llm_response)
    if not isinstance(analysis_results, dict) or not analysis_results:
        raise ValueError("Blueprint regeneration returned empty analysis results")
    
    return analysis_results


def get_user_feedback(
    logger,
    iteration: int,
    verification_results: Optional[Dict[str, Any]] = None,
    simulation_results: Optional[Dict[str, Any]] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prompt user for manual feedback input, showing current iteration summary first.
    """
    # Check if running in an interactive terminal
    import sys
    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    
    if not is_interactive:
        logger.warning("=" * 80)
        logger.warning("WARNING: Running in non-interactive mode - cannot prompt for user feedback")
        logger.warning("=" * 80)
        logger.warning("The program is running in a non-interactive environment (e.g., background process, redirected output).")
        logger.warning("User feedback input is disabled. Using system-generated feedback only.")
        logger.warning("=" * 80)
        logger.warning("To enable user feedback, run the script in an interactive terminal.")
        logger.warning("=" * 80)
        return ""
    
    # First show the iteration summary
    display_iteration_summary(logger, iteration, verification_results, simulation_results, evaluation_results, generated_code)
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    sys.stderr.flush()
    
    print("\n" + "="*80)
    print("MANUAL FEEDBACK INPUT")
    print("="*80)
    print("Based on the execution summary above, please provide your feedback for the current iteration.")
    print("This feedback will be used to improve the simulation code in the next iteration.")
    print("You can include suggestions for:")
    print("- Code improvements or bug fixes")
    print("- Model accuracy enhancements")
    print("- Performance optimizations")
    print("- Any other observations or recommendations")
    print("\n⚠️  ITERATION CONTROL:")
    print("- If you want to STOP iterations and finalize results, type: #STOP#")
    print("- Otherwise, the system will continue to the next iteration after your feedback")
    print("\nIf you don't want to provide feedback, just press Enter twice to skip.")
    print("Otherwise, enter your feedback (press Enter twice to finish):")
    print("-"*80)
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    
    feedback_lines = []
    empty_line_count = 0
    
    try:
        while True:
            try:
                line = input()
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    feedback_lines.append(line)
                else:
                    empty_line_count = 0
                    feedback_lines.append(line)
            except EOFError:
                # EOFError means no input is available (non-interactive environment)
                logger.warning("EOFError: No input available - non-interactive environment detected")
                logger.warning("Skipping user feedback input. Using system-generated feedback only.")
                break
            except KeyboardInterrupt:
                print("\nFeedback input interrupted by user.")
                logger.info("User interrupted feedback input")
                break
    except Exception as e:
        logger.error(f"Unexpected error during feedback input: {e}")
        logger.error("Skipping user feedback input. Using system-generated feedback only.")
    
    user_feedback = "\n".join(feedback_lines).strip()
    
    if user_feedback:
        print("-"*80)
        print("Your feedback has been recorded:")
        print(user_feedback)
        print("="*80)
        logger.info(f"User feedback received: {len(user_feedback)} characters")
    else:
        print("-"*80)
        print("No feedback provided. Using system-generated feedback only.")
        print("="*80)
        logger.info("No user feedback provided - using system feedback only")
    
    return user_feedback

def display_iteration_summary(
    logger,
    iteration: int,
    verification_results: Optional[Dict[str, Any]] = None,
    simulation_results: Optional[Dict[str, Any]] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, Any]] = None
):
    """Display a summary of the current iteration's execution status."""
    print("\n" + "="*80)
    print(f"ITERATION {iteration + 1} EXECUTION SUMMARY")
    print("="*80)
    
    # Code Generation Summary
    print("📝 CODE GENERATION:")
    if generated_code:
        if "metadata" in generated_code:
            metadata = generated_code["metadata"]
            print(f"   ✓ Model Type: {metadata.get('model_type', 'Unknown')}")
            if 'entities' in metadata:
                print(f"   ✓ Entities: {', '.join(metadata['entities'])}")
            if 'behaviors' in metadata:
                print(f"   ✓ Behaviors: {', '.join(metadata['behaviors'])}")
        
        code_summary = generated_code.get("code_summary", "No summary available")
        print(f"   ✓ Summary: {code_summary}")
    else:
        print("   ❌ No code generation results available")
    
    # Code Verification Summary
    print("\n🔍 CODE VERIFICATION:")
    if verification_results:
        if verification_results.get("passed", False):
            print("   ✅ Status: PASSED")
            print(f"   ✓ Summary: {verification_results.get('summary', 'Verification successful')}")
        else:
            print("   ❌ Status: FAILED")
            if "critical_issues" in verification_results:
                print("   ❌ Critical Issues:")
                for issue in verification_results["critical_issues"]:
                    print(f"      • {issue}")
            if "warnings" in verification_results:
                print("   ⚠️  Warnings:")
                for warning in verification_results["warnings"]:
                    print(f"      • {warning}")
            print(f"   📋 Summary: {verification_results.get('summary', 'Verification failed')}")
    else:
        print("   ❓ No verification results available")
    
    # Simulation Execution Summary
    print("\n🚀 SIMULATION EXECUTION:")
    if simulation_results:
        execution_status = simulation_results.get("execution_status", "unknown")
        if execution_status == "success":
            print("   ✅ Status: SUCCESS")
            
            # Performance metrics
            if "performance_metrics" in simulation_results:
                perf = simulation_results["performance_metrics"]
                if "execution_time" in perf:
                    print(f"   ⏱️  Execution Time: {perf['execution_time']:.2f} seconds")
                if "memory_usage" in perf:
                    print(f"   💾 Memory Usage: {perf['memory_usage']} MB")
            
            # Simulation metrics
            if "simulation_metrics" in simulation_results:
                sim_metrics = simulation_results["simulation_metrics"]
                print("   📊 Simulation Metrics:")
                for key, value in sim_metrics.items():
                    print(f"      • {key}: {value}")
            
            # Time series data summary
            if "time_series_data" in simulation_results:
                ts_data = simulation_results["time_series_data"]
                if ts_data:
                    print(f"   📈 Time Series: {len(ts_data)} data points collected")
            
        elif execution_status == "failed":
            print("   ❌ Status: FAILED")
            if "runtime_errors" in simulation_results:
                print("   ❌ Runtime Errors:")
                for error in simulation_results["runtime_errors"]:
                    # Truncate very long error messages
                    error_str = str(error)
                    if len(error_str) > 200:
                        error_str = error_str[:200] + "..."
                    print(f"      • {error_str}")
        else:
            print(f"   ❓ Status: {execution_status.upper()}")
        
        summary = simulation_results.get("summary", "No summary available")
        print(f"   📋 Summary: {summary}")
    else:
        print("   ❓ No simulation execution results available")
    
    # Result Evaluation Summary
    print("\n📊 RESULT EVALUATION:")
    if evaluation_results:
        if evaluation_results.get("placeholder"):
            print("   ⚠️  No real evaluation data available (placeholder).")
        else:
            if "overall_score" in evaluation_results:
                score = evaluation_results["overall_score"]
                print(f"   📈 Overall Score: {score}")

            if "metrics" in evaluation_results:
                metrics = evaluation_results["metrics"]
                print("   📋 Evaluation Metrics:")
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            print(f"      • {metric_name}: {metric_value:.4f}")
                        else:
                            print(f"      • {metric_name}: {metric_value}")
                elif isinstance(metrics, list):
                    for i, metric in enumerate(metrics):
                        if isinstance(metric, dict):
                            for key, value in metric.items():
                                print(f"      • {key}: {value}")
                        else:
                            print(f"      • Metric {i+1}: {metric}")
                else:
                    print(f"      • {metrics}")

            if "recommendations" in evaluation_results:
                recommendations = evaluation_results["recommendations"]
                if recommendations:
                    print("   💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show only first 3 recommendations
                        print(f"      • {rec}")
    else:
        print("   ❓ No evaluation results available")
    
    print("="*80)

@inject
def run_data_analysis_test(
    args,
    logger,
    agent_container: AgentContainer = Provide[AgentContainer]
):
    """Run the data analysis test with multi-iteration support (ODD mode + lite workflow)."""
    
    logger.info(f"Starting DataAnalysisOddAgent test in {args.mode.upper()} mode")
    logger.info(f"Auto mode: {args.auto}, Max iterations: {args.iterations}")
    
    try:
        # Set up output path in container
        agent_container.output_path.override(args.output)
        
        # Get agent instances - include all agents needed for iterative workflow
        agents = {
            "data_analysis": agent_container.data_analysis_odd_agent(),
            "code_generation": agent_container.code_generation_odd_agent(),
            "code_verification": agent_container.code_verification_agent(),
            "simulation_execution": agent_container.simulation_execution_agent(),
            "result_evaluation": agent_container.result_evaluation_agent(),
            "feedback_generation": agent_container.feedback_generation_odd_agent(),  # Use odd agent for odd mode
            "iteration_control": agent_container.iteration_control_agent()
        }
        
        # Initialize state management (similar to workflow_manager)
        state = {
            "task_spec": None,
            "data_analysis": None,
            "model_plan": None,
            "generated_code": None,
            "verification_results": None,
            "simulation_results": None,
            "evaluation_results": None,
            "feedback": None,
            "iteration_decision": None
        }
        
        # Initialize code memory to store generated code per iteration
        code_memory = {}
        state["code_memory"] = code_memory
        
        # Initialize historical fix log
        historical_fix_log = {}
        
        # Initialize current iteration counter
        current_iteration = 0
        
        # Variables to track if we're skipping initial phases
        skip_data_analysis = False
        skip_initial_code_generation = False
        
        # ==================================================
        # PHASE 1: Data Analysis or Load Persisted Data
        # ==================================================
        
        # Check if we should load persisted code (skip data analysis AND initial code generation)
        if hasattr(args, 'persisted_code_file') and getattr(args, 'persisted_code_file', None):
            logger.info("=" * 50)
            logger.info("LOADING PERSISTED CODE")
            logger.info("=" * 50)
            
            # Load the persisted code
            persisted_code = load_persisted_code(args.persisted_code_file)
            
            # Also need to load task_spec
            if hasattr(args, 'persisted_data_analysis_file') and getattr(args, 'persisted_data_analysis_file', None):
                task_spec = load_persisted_data_analysis(args.persisted_data_analysis_file)
            else:
                # Try to find task_spec in the same directory as persisted code
                code_dir = os.path.dirname(args.persisted_code_file)
                task_spec_file = os.path.join(code_dir, "task_spec_iter_0.json")
                if os.path.exists(task_spec_file):
                    task_spec = load_persisted_data_analysis(task_spec_file)
                    logger.info(f"Auto-loaded task_spec from: {task_spec_file}")
                else:
                    logger.error("Cannot find task_spec file. Please provide --persisted-data-analysis-file")
                    return {"status": "failed", "error": "Missing task_spec", "output_path": args.output}
            
            # Extract data path
            data_path = None
            if "data_folder" in task_spec:
                data_folder = task_spec["data_folder"]
                if os.path.isabs(data_folder):
                    data_path = data_folder
                else:
                    data_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
                logger.info(f"Data path from persisted file: {data_path}")
            
            # Create initial generated_code dict from persisted code
            state["generated_code"] = {
                "code": persisted_code,
                "code_summary": f"Loaded persisted code ({len(persisted_code)} characters)",
                "metadata": {
                    "model_type": "odd",
                    "mode": "odd",
                    "source": "persisted"
                }
            }
            
            # Store in code_memory as iter_0 (the persisted code is iter_0)
            code_memory[0] = {f"simulation_code_iter_0.py": persisted_code}
            
            # Save the loaded code as iter_0 (if it doesn't exist already)
            # Check if iter_0 already exists, if not, save it
            iter_0_path = os.path.join(args.output, "simulation_code_iter_0.py")
            if not os.path.exists(iter_0_path):
                save_generated_code(args.output, state["generated_code"], iteration=0)
            
            skip_data_analysis = True
            skip_initial_code_generation = True
            current_iteration = 0  # Start from iteration 0 since we loaded iter_0, will generate feedback first, then iter_1
            
            logger.info("Persisted code loaded successfully as iter_0, will generate feedback first, then iter_1")
            
        elif hasattr(args, 'persisted_data_analysis_file') and getattr(args, 'persisted_data_analysis_file', None):
            # Skip only data analysis, but still do initial code generation
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
                    data_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
                logger.info(f"Data path from persisted file: {data_path}")
                
            skip_data_analysis = True
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
                    task_data=task_data,
                    mode=args.mode
                )
            else:
                task_spec = agents["data_analysis"].process(
                    task_description=args.task,
                    mode=args.mode
                )
        
        logger.info("Data analysis and task spec generation completed successfully")
        
        # Save task_spec (after all branches)
        state["task_spec"] = task_spec
        if not skip_data_analysis or current_iteration == 0:
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
            
        # ==================================================
        # MAIN ITERATION LOOP (Lite-style workflow)
        # ==================================================
        
        while current_iteration < args.iterations:
            logger.info("=" * 50)
            logger.info(f"STARTING ITERATION {current_iteration + 1}/{args.iterations}")
            logger.info("=" * 50)
        
            # --------------------------------------------------
            # STEP 1: Code Generation
            # --------------------------------------------------
            if not skip_initial_code_generation or current_iteration > 0:
                logger.info("CODE GENERATION")
                
                # Get previous code if exists
                prev_code = None
                if current_iteration > 0 and current_iteration - 1 in code_memory:
                    prev_code_dict = code_memory[current_iteration - 1]
                    prev_code_filename = f"simulation_code_iter_{current_iteration - 1}.py"
                    if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                        prev_code = prev_code_dict[prev_code_filename]
                    elif isinstance(prev_code_dict, str):
                        prev_code = prev_code_dict
                
                # Generate code
                state["generated_code"] = agents["code_generation"].process(
            task_spec=task_spec,
            data_analysis=None,  # Not used in odd mode
            model_plan=None,     # Not used in odd mode
                    feedback=state["feedback"],  # Will be None in first iteration
            data_path=data_path,
                    previous_code=prev_code,
                    historical_fix_log=historical_fix_log,
                    mode=args.mode,
                    selfloop=args.selfloop,
                    blueprint=None,
                    output_dir=args.output,
                    iteration=current_iteration
                )
                
                # Save generated code (use current_iteration as file number: iter_0, iter_1, etc.)
                save_generated_code(args.output, state["generated_code"], iteration=current_iteration)
                
                # Store in code_memory (key: current_iteration, value: {filename: code})
                gen_code_dict = state["generated_code"]["code"]
                code_memory[current_iteration] = {f"simulation_code_iter_{current_iteration}.py": gen_code_dict}
                
                logger.info(f"Code generation completed for iteration {current_iteration} (saved as iter_{current_iteration})")
            else:
                logger.info("Skipping initial code generation (using persisted code)")
                skip_initial_code_generation = False  # Reset for next iteration
            
            # --------------------------------------------------
            # STEP 2: Code Verification
            # --------------------------------------------------
            # Determine code file path based on whether code was generated or loaded
            if skip_initial_code_generation and current_iteration == 0:
                # Using persisted code (iter_0)
                code_file_path = os.path.join(args.output, f"simulation_code_iter_0.py")
            else:
                # Using generated code (iter_{current_iteration})
                code_file_path = os.path.join(args.output, f"simulation_code_iter_{current_iteration}.py")
            
            # ODD mode: Skip verification, simulation, and evaluation, use placeholders
            if args.mode == "odd":
                logger.info("ODD mode: Skipping verification, simulation, and evaluation - using placeholders")
                state["verification_results"] = {
                    "placeholder": True,
                    "note": "Verification is not executed in ODD mode",
                    "passed": True
                }
                state["simulation_results"] = {
                    "placeholder": True,
                    "note": "Simulation execution is not performed in ODD mode",
                    "execution_status": "skipped"
                }
                state["evaluation_results"] = {
                    "placeholder": True,
                    "note": "Evaluation is not performed in ODD mode"
                }
                save_artifact(args.output, f"verification_results_iter_{current_iteration}", state["verification_results"])
                save_artifact(args.output, f"simulation_results_iter_{current_iteration}", state["simulation_results"])
                save_artifact(args.output, f"evaluation_results_iter_{current_iteration}", state["evaluation_results"])
                logger.info("ODD mode: Placeholders created for verification, simulation, and evaluation results")
            else:
                # Full mode: Execute verification, simulation, and evaluation
                logger.info("CODE VERIFICATION")
                # Get code from generated_code or code_memory
                if state.get("generated_code") and state["generated_code"].get("code"):
                    code_to_verify = state["generated_code"]["code"]
                elif current_iteration in code_memory:
                    code_dict = code_memory[current_iteration]
                    code_filename = f"simulation_code_iter_{current_iteration}.py"
                    code_to_verify = code_dict.get(code_filename) if isinstance(code_dict, dict) else code_dict
                else:
                    logger.warning("No code available for verification")
                    code_to_verify = ""
                
                state["verification_results"] = agents["code_verification"].process(
                    code=code_to_verify,
                    task_spec=task_spec,
                    data_path=data_path,
                    use_sandbox=False  # Lightweight verification
                )
                save_artifact(args.output, f"verification_results_iter_{current_iteration}", state["verification_results"])
                
                if state["verification_results"]["passed"]:
                    logger.info(f"✅ Code verification PASSED for iteration {current_iteration}")
                else:
                    logger.warning(f"❌ Code verification FAILED for iteration {current_iteration}")
                
                # --------------------------------------------------
                # STEP 3 & 4: Simulation Execution & Evaluation
                # --------------------------------------------------
                if state["verification_results"]["passed"]:
                    logger.info("SIMULATION EXECUTION")
                    state["simulation_results"] = agents["simulation_execution"].process(
                        code_path=code_file_path,
                        task_spec=task_spec,
                        data_path=data_path,
                        mode="lite"  # Use lite mode for subprocess execution
                    )
                    save_artifact(args.output, f"simulation_results_iter_{current_iteration}", state["simulation_results"])
                    
                    if state["simulation_results"] and state["simulation_results"].get("execution_status") == "success":
                        logger.info(f"✅ Simulation execution completed successfully")
                    else:
                        logger.warning(f"❌ Simulation execution failed")
                    
                    logger.info("RESULT EVALUATION")
                    state["evaluation_results"] = agents["result_evaluation"].process(
                        simulation_results=state["simulation_results"],
                        task_spec=task_spec,
                        data_analysis=None  # No data analysis in odd mode
                    )
                    save_artifact(args.output, f"evaluation_results_iter_{current_iteration}", state["evaluation_results"])
                else:
                    logger.warning("Skipping execution and evaluation due to verification failure")
                    state["simulation_results"] = None
                    state["evaluation_results"] = None
            
            # --------------------------------------------------
            # STEP 5: Feedback Generation
            # --------------------------------------------------
            logger.info("FEEDBACK GENERATION")
            
            # Get current and previous code for feedback
            # If we skipped initial code generation (loading persisted code), use iter_0
            # Otherwise, use the generated code from current iteration
            if skip_initial_code_generation and current_iteration == 0:
                # Loading persisted code as iter_0, use iter_0 for feedback
                current_code_dict = code_memory[current_iteration]
                current_code_filename = f"simulation_code_iter_0.py"
                current_code = current_code_dict[current_code_filename]
                previous_code = None  # No previous code for iter_0
            else:
                # Normal flow: use generated code from current iteration
                current_code_dict = code_memory[current_iteration]
                current_code_filename = f"simulation_code_iter_{current_iteration}.py"
                current_code = current_code_dict[current_code_filename]
                
                previous_code = None
                if current_iteration > 0 and current_iteration - 1 in code_memory:
                    prev_code_dict = code_memory[current_iteration - 1]
                    prev_code_filename = f"simulation_code_iter_{current_iteration - 1}.py"
                    if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                        previous_code = prev_code_dict[prev_code_filename]
                    elif isinstance(prev_code_dict, str):
                        previous_code = prev_code_dict
            
            # Collect manual feedback first (odd-mode requirement)
            user_feedback_text = None
            if not args.auto:
                logger.info("Manual feedback mode - prompting user for feedback before LLM generation")
                user_feedback_text = get_user_feedback(
                    logger,
                    current_iteration,
                    verification_results=state["verification_results"],
                    simulation_results=state["simulation_results"],
                    evaluation_results=state["evaluation_results"],
                    generated_code=state["generated_code"]
                )
            
            # Generate system feedback (LLM)
            system_feedback = agents["feedback_generation"].process(
                task_spec=task_spec,
                model_plan=None,  # Not used in odd mode
                generated_code=state["generated_code"],
                verification_results=state["verification_results"],
                simulation_results=state["simulation_results"],
                evaluation_results=state["evaluation_results"],
                current_code=current_code,
                previous_code=previous_code,
                iteration=current_iteration,
                historical_fix_log=historical_fix_log,
                mode=args.mode  # Pass mode parameter for odd mode detection
            )
            
            # Combine user feedback (if any) with system feedback
            if not args.auto:
                combined_feedback = dict(system_feedback)
                
                if user_feedback_text:
                    # Create user feedback section
                    user_feedback_section = {
                        "source": "user",
                        "content": user_feedback_text,
                        "note": "This is user-provided feedback. Please pay special attention to these suggestions."
                    }
                    
                    # Add user feedback to the combined feedback structure
                    if "feedback_sections" not in combined_feedback:
                        combined_feedback["feedback_sections"] = []
                    
                    # Insert user feedback at the beginning to prioritize it
                    combined_feedback["feedback_sections"].insert(0, {
                        "section": "USER_FEEDBACK",
                        "priority": "CRITICAL",
                        "feedback": user_feedback_section
                    })
                    
                    # Also add to summary
                    if "summary" in combined_feedback:
                        combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}\n\nSYSTEM FEEDBACK: {combined_feedback['summary']}"
                    else:
                        combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}"
                    
                    logger.info("User feedback has been integrated with system feedback")
                else:
                    logger.info("No user feedback provided - using system feedback only")
                
                state["feedback"] = combined_feedback
            else:
                # Auto mode - use system feedback only
                state["feedback"] = system_feedback
            
            save_artifact(args.output, f"feedback_iter_{current_iteration}", state["feedback"])
            
            # Reset skip_initial_code_generation after first iteration (when loading persisted code)
            # This ensures that subsequent iterations will generate code normally
            if skip_initial_code_generation and current_iteration == 0:
                skip_initial_code_generation = False
                logger.info("Reset skip_initial_code_generation flag - next iteration will generate code")
            
            # --------------------------------------------------
            # STEP 6: Iteration Control Decision
            # --------------------------------------------------
            logger.info("ITERATION CONTROL DECISION")
            
            # Extract user feedback if available for stop command check
            user_feedback_text = None
            if not args.auto and state["feedback"]:
                feedback_sections = state["feedback"].get("feedback_sections", [])
                for section in feedback_sections:
                    if section.get("section") == "USER_FEEDBACK":
                        user_feedback_text = section.get("feedback", {}).get("content", "")
                        break
            
            state["iteration_decision"] = agents["iteration_control"].process(
                feedback=state["feedback"],
                verification_results=state["verification_results"],
                evaluation_results=state["evaluation_results"],
                current_iteration=current_iteration,
                max_iterations=args.iterations,
                auto_mode=args.auto,
                user_feedback=user_feedback_text
            )
            save_artifact(args.output, f"iteration_decision_iter_{current_iteration}", state["iteration_decision"])
            
            # Check if we should continue
            if not state["iteration_decision"]["continue"]:
                logger.info(f"🛑 Stopping after {current_iteration + 1} iterations: {state['iteration_decision']['reason']}")
                break
            
            # Move to next iteration
            current_iteration += 1
        
        # ==================================================
        # FINAL SUMMARY
        # ==================================================
        # Note: After loop ends, current_iteration is the next iteration number
        # So the last completed iteration is current_iteration - 1
        # But if we completed N iterations (0 to N-1), current_iteration = N
        final_iteration = current_iteration  # This is the total number of iterations completed
        last_code_iteration = current_iteration - 1 if current_iteration > 0 else 0
        
        logger.info("=" * 50)
        logger.info("TEST COMPLETION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Data Analysis: Completed")
        logger.info(f"✅ Code Generation: Completed ({final_iteration} iterations)")
        logger.info(f"📁 Artifacts saved to: {args.output}")
        logger.info(f"📄 Final code: simulation_code_iter_{last_code_iteration}.py")
        
        return {
            "status": "success",
            "state": state,
            "task_spec": task_spec,
            "data_analysis": data_analysis_result,
            "generated_code": state["generated_code"],
            "total_iterations": final_iteration,
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