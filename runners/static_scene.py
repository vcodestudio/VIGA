#!/usr/bin/env python3
"""
Static Scene Runner for AgenticVerifier
Loads static scene dataset and runs the dual-agent system for 3D static scene generation from scratch.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_model_info, get_meshy_info

def load_static_scene_dataset(base_path: str, task_name: str, setting: str, test_id: Optional[str] = None) -> List[Dict]:
    """
    Load static scene dataset structure.

    Args:
        base_path: Path to static scene dataset root.
        task_name: Task name to load.
        setting: Initialization setting for the task.
        test_id: Optional test ID for filtering.

    Returns:
        List of task configurations.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Error: Static scene dataset path not found: {base_path}")
        return []
    
    tasks = []
    
    # For static scenes, we typically have target images and descriptions
    if task_name == "all":
        # Load all available static scene tasks
        task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        task_list = [d.name for d in task_dirs]
    else:
        task_list = [task_name]
    
    # Create task configurations
    for task in task_list:
        task_path = base_path / task
        if not task_path.exists():
            print(f"Warning: Task directory not found: {task_path}")
            continue
        
        print("task_path:", task_path)
            
        # Look for target images
        target_image_path = None
        if (task_path / "target.png").exists():
            target_image_path = str(task_path / "target.png")
        elif (task_path / "target.jpg").exists():
            target_image_path = str(task_path / "target.jpg")
        elif (task_path / "target.jpeg").exists():
            target_image_path = str(task_path / "target.jpeg")
        elif (task_path / "target").exists() and (task_path / "target").is_dir():
            target_image_path = str(task_path / "target")
        
        if not target_image_path:
            print(f"Warning: No target image found for task: {task}")
            continue
        
        init_image_path = task_path / f"{setting}_init" / "render1.png"
        
        if not init_image_path.exists():
            init_image_path = ''
        
        # Look for description file
        description_path = task_path / "description.txt"
        target_description = None
        if description_path.exists():
            with open(description_path, 'r') as f:
                target_description = f.read().strip()
        
        # Check for assets directory
        assets_path = task_path / "assets"
        os.makedirs(assets_path, exist_ok=True)
        assets_dir = str(assets_path)
        
        task_config = {
            "task_name": task,
            "task_id": task,
            "target_image_path": target_image_path,
            "assets_dir": assets_dir,  # Add assets directory path
            "output_dir": f"output/static_scene/{test_id or time.strftime('%Y%m%d_%H%M%S')}/{task}",
            "init_code_path": "",  # Static scenes start from scratch
            "init_image_path": str(init_image_path),
        }
        
        tasks.append(task_config)
    
    return tasks


def _rewind_to_round(resume_dir: Path, target_round: int) -> None:
    """Rewind output directory to a specific round.

    Deletes all renders, scripts, generator/verifier thoughts AFTER target_round.
    Truncates generator_memory.json to only include up to target_round.

    Args:
        resume_dir: Path to the task output directory.
        target_round: The round to resume from (data for this round is kept, everything after is deleted).
    """
    print(f"\n=== Rewinding to round {target_round} ===")

    # 1. Delete renders/N for N > target_round
    renders_dir = resume_dir / "renders"
    if renders_dir.exists():
        for subdir in renders_dir.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                round_num = int(subdir.name)
                if round_num > target_round:
                    print(f"  Deleting renders/{subdir.name}/")
                    shutil.rmtree(subdir, ignore_errors=True)

    # 2. Delete scripts/N.py for N > target_round
    scripts_dir = resume_dir / "scripts"
    if scripts_dir.exists():
        for f in scripts_dir.iterdir():
            if f.is_file() and f.stem.isdigit():
                round_num = int(f.stem)
                if round_num > target_round:
                    print(f"  Deleting scripts/{f.name}")
                    f.unlink(missing_ok=True)

    # 3. Delete generator_thoughts/N.* for N > target_round
    for thoughts_dir_name in ["generator_thoughts", "verifier_thoughts"]:
        thoughts_dir = resume_dir / thoughts_dir_name
        if thoughts_dir.exists():
            for f in thoughts_dir.iterdir():
                if f.is_file() and f.stem.isdigit():
                    round_num = int(f.stem)
                    if round_num > target_round:
                        print(f"  Deleting {thoughts_dir_name}/{f.name}")
                        f.unlink(missing_ok=True)

    # 4. Delete codes/N.* for N > target_round
    codes_dir = resume_dir / "codes"
    if codes_dir.exists():
        for f in codes_dir.iterdir():
            if f.is_file() and f.stem.isdigit():
                round_num = int(f.stem)
                if round_num > target_round:
                    print(f"  Deleting codes/{f.name}")
                    f.unlink(missing_ok=True)

    # 5. Truncate generator_memory.json
    #    Round counting: round 0 starts from the beginning.
    #    Each round adds messages to the memory.
    #    We count rounds by looking at verifier_result patterns and tool responses.
    #    Strategy: count "user" messages that contain verifier results (render feedback)
    #    as round boundaries. The initial user message (round 0) is the first one after system.
    memory_path = resume_dir / "generator_memory.json"
    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)

        # Count rounds by tracking assistant→tool pairs
        # Round 0 = first assistant response, etc.
        # Each round consists of: assistant message → tool response(s) → (optional verifier user msg)
        # We identify round boundaries by counting sequences of (assistant, tool*) groups
        round_count = -1  # Start at -1, first assistant increments to 0
        keep_until_idx = len(memory)  # Default: keep all

        i = 0
        while i < len(memory):
            msg = memory[i]
            if msg.get("role") == "assistant" and i > 0:  # Skip system-level messages
                round_count += 1
                if round_count > target_round:
                    # Found the first message of a round we want to delete
                    keep_until_idx = i
                    break
            i += 1

        if keep_until_idx < len(memory):
            deleted_count = len(memory) - keep_until_idx
            memory = memory[:keep_until_idx]
            with open(memory_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=4, ensure_ascii=False)
            print(f"  Truncated generator_memory.json: kept {keep_until_idx} messages, deleted {deleted_count}")
        else:
            print(f"  generator_memory.json: target_round {target_round} >= total rounds ({round_count + 1}), no truncation needed")

    print(f"=== Rewind to round {target_round} complete ===\n")


def run_static_scene_task(task_config: Dict, args: argparse.Namespace) -> Tuple[str, bool, Optional[str]]:
    """
    Run a single static scene task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
    
    Returns:
        Tuple of (task_name, success, error_message)
    """
    task_name = task_config["task_name"]
    is_resume = hasattr(args, 'resume_path') and args.resume_path
    resume_round = getattr(args, 'resume_round', None)
    
    if is_resume:
        # Resume mode: use the existing directory directly (no copying)
        resume_dir = Path(args.resume_path)
        task_config["output_dir"] = str(resume_dir)  # Use existing directory
        print(f"Resuming static scene task: {task_name} in {resume_dir}")

        # If resume_round is set, rewind to that round first
        if resume_round is not None:
            _rewind_to_round(resume_dir, resume_round)
        
        # Find the latest state.blend in renders folder to use as starting point
        renders_dir = resume_dir / "renders"
        existing_blend = None
        if renders_dir.exists():
            render_subdirs = sorted([d for d in renders_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else 0, reverse=True)
            for subdir in render_subdirs:
                state_blend = subdir / "state.blend"
                if state_blend.exists():
                    existing_blend = str(state_blend)
                    break
        
        if existing_blend:
            # Copy latest state.blend to blender_file.blend for continuation
            created_blender_file = str(resume_dir / "blender_file.blend")
            shutil.copy(existing_blend, created_blender_file)
            print(f"  Using latest state: {existing_blend}")
        else:
            # Use existing blender_file.blend
            created_blender_file = str(resume_dir / "blender_file.blend")
            if not Path(created_blender_file).exists():
                print(f"  Warning: No blend file found in {resume_dir}, starting fresh")
                is_resume = False
            else:
                print(f"  Using existing blend file: {created_blender_file}")
    else:
        print(f"Running static scene task: {task_name}")
    
    # Create output directory (only for new runs)
    os.makedirs(task_config["output_dir"], exist_ok=True)
    
    if not is_resume:
        # Create an empty blender file inside output_dir for build-from-scratch flows
        created_blender_file = os.path.join(task_config["output_dir"], "blender_file.blend")
        # copy the blender file to the output directory
        if os.path.exists(args.blender_file):
            shutil.copy(args.blender_file, created_blender_file)
        else:
            # Create a new blender file - use forward slashes for cross-platform compatibility
            created_blender_file_escaped = created_blender_file.replace("\\", "/")
            create_empty_blend_cmd = (
                f'"{args.blender_command}" --background --factory-startup '
                f'--python-expr "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.wm.save_mainfile(filepath=\'{created_blender_file_escaped}\')"'
            )
            subprocess.run(create_empty_blend_cmd, shell=True, check=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "static_scene",
        "--model", args.model,
        "--api-key", get_model_info(args.model)["api_key"],
        "--api-base-url", get_model_info(args.model)["base_url"],
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--target-image-path", task_config["target_image_path"] if not args.text_only else "",
        "--output-dir", task_config["output_dir"],
        "--task-name", task_name,
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        "--blender-command", args.blender_command,
        "--blender-file", created_blender_file,
        "--blender-script", args.blender_script,
        "--meshy_api_key", get_meshy_info()["meshy_api_key"],
        "--va_api_key", get_meshy_info()["va_api_key"],
        "--blender-save", created_blender_file,
        "--assets-dir", task_config["assets_dir"],
        "--init-code-path", task_config["init_code_path"],
        "--init-image-path", task_config["init_image_path"],
        "--prompt-setting", args.prompt_setting,
        "--init-setting", args.init_setting,
        "--render-engine", args.render_engine,
        "--effect", args.effect,
    ]
    
    # Add resume flag or clear-memory flag
    memory_path = Path(task_config["output_dir"]) / "generator_memory.json"
    if is_resume and memory_path.exists():
        cmd.extend(["--resume-memory", str(memory_path)])
    else:
        cmd.extend(["--clear-memory"])
    
    if args.explicit_comp:
        cmd.extend(["--explicit-comp"])
    if args.gpu_devices:
        cmd.extend(["--gpu-devices", args.gpu_devices])
    if "target_description" in task_config:
        cmd.extend(["--target-description", task_config["target_description"]])

    try:
        result = subprocess.run(cmd)  # no timeout
        
        if result.returncode == 0:
            print(f"Static scene task {task_name} completed successfully")
            return task_name, True, None
        else:
            error_msg = f"Task failed with return code {result.returncode}: {result.stderr}"
            print(f"Static scene task {task_name} failed: {error_msg}")
            return task_name, False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out after 1 hour"
        print(f"Static scene task {task_name} timed out")
        return task_name, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Static scene task {task_name} failed with exception: {error_msg}")
        return task_name, False, error_msg


def run_static_scene_tasks_parallel(tasks: List[Dict], args: argparse.Namespace, max_workers: int = 4) -> None:
    """Run static scene tasks in parallel.

    Args:
        tasks: List of task configurations.
        args: Command line arguments.
        max_workers: Maximum number of parallel workers.
    """
    print(f"Running {len(tasks)} static scene tasks with {max_workers} workers")
    
    successful_tasks = 0
    failed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_static_scene_task, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task_config = future_to_task[future]
            try:
                task_name, success, error_msg = future.result()
                if success:
                    successful_tasks += 1
                else:
                    failed_tasks += 1
                    print(f"Failed task: {task_name} - {error_msg}")
            except Exception as e:
                failed_tasks += 1
                print(f"Task {task_config['task_name']} generated an exception: {e}")
    
    print(f"\nStatic scene task execution completed:")
    print(f"  Successful: {successful_tasks}")
    print(f"  Failed: {failed_tasks}")
    print(f"  Total: {len(tasks)}")


def main() -> None:
    """Entry point for the static scene runner."""
    parser = argparse.ArgumentParser(description="Static Scene Runner for AgenticVerifier")
    time_str = time.strftime('%Y%m%d_%H%M%S')
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", "data/static_scene"), help="Path to static scene dataset root directory")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_ROOT", f"output/static_scene/{time_str}"), help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", default=os.getenv("TASK", "all"), help="Specific task to run (default: all)")
    parser.add_argument("--test-id", help="Test ID for output directory naming")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=int(os.getenv("MAX_ROUNDS", "100")), help="Maximum number of interaction rounds")
    parser.add_argument("--model", default=os.getenv("MODEL", "gemini-3-flash-preview"), help="OpenAI vision model to use")
    parser.add_argument("--memory-length", type=int, default=int(os.getenv("MEMORY_LENGTH", "12")), help="Memory length")
    
    # Blender parameters
    parser.add_argument("--blender-command", default=os.getenv("BLENDER_COMMAND", "/Applications/Blender.app/Contents/MacOS/Blender"), help="Blender command path")
    parser.add_argument("--blender-file", default="data/static_scene/empty_scene.blend", help="Empty blender file for static scenes")
    parser.add_argument("--blender-script", default="data/static_scene/generator_script.py", help="Blender execution script")
    parser.add_argument("--blender-save", default=f"data/static_scene/empty_scene.blend", help="Save blender file")
    
    # Tool server scripts (comma-separated)
    # Build default generator tools based on segment-objects (SAM + ComfyUI) setting
    default_generator_tools = "tools/blender/exec.py,tools/generator_base.py,tools/initialize_plan.py"
    if os.getenv("SEGMENT_OBJECTS", "false").lower() == "true":
        default_generator_tools += ",tools/sam3d/init.py"
    parser.add_argument("--generator-tools", default=default_generator_tools, help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--verifier-tools", default="tools/blender/investigator.py,tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    # Additional parameters
    parser.add_argument("--explicit-comp", action="store_true", help="Enable explicit completion")
    parser.add_argument("--text-only", action="store_true", help="Only use text as reference")
    parser.add_argument("--init-setting", choices=["none", "minimal", "reasonable"], default="none", help="Setting for the static scene task")
    parser.add_argument("--prompt-setting", choices=["none", "procedural", "scene_graph", "get_asset", "segment_objects"], default="none", help="Setting for the prompt (segment_objects: SAM segment + ComfyUI when SEGMENT_OBJECTS=true)")
    parser.add_argument("--render-engine", default=os.getenv("RENDER_ENGINE", "eevee"), choices=["eevee", "cycles", "workbench", "solid", "outline"], help="Render engine (eevee=fast, cycles=quality, workbench=fastest, solid=solid view). Default: eevee")
    parser.add_argument("--effect", default=os.getenv("RENDER_EFFECT", "none"), choices=["none", "freestyle"], help="Render effect (none=default, freestyle=line art). Default: none")
    parser.add_argument("--resume", default=os.getenv("RESUME", "false").lower() == "true", action="store_true", help="Enable resume mode")
    resume_path_env = os.getenv("RESUME_PATH", None)
    # Treat 'null', 'none', empty string as None
    if resume_path_env and resume_path_env.lower() in ('null', 'none', ''):
        resume_path_env = None
    parser.add_argument("--resume-path", default=resume_path_env, help="Path to resume from. If not set with --resume, auto-detects latest run.")
    resume_round_env = os.getenv("RESUME_ROUND", None)
    if resume_round_env and resume_round_env.lower() in ('null', 'none', ''):
        resume_round_env = None
    parser.add_argument("--resume-round", type=int, default=int(resume_round_env) if resume_round_env else None,
                        help="Resume from specific round (delete rounds after this and restart from here)")
    
    args = parser.parse_args()

    # When SEGMENT_OBJECTS=true, default to segment_objects prompt (initialize + get_better_object via ComfyUI)
    if os.getenv("SEGMENT_OBJECTS", "false").lower() == "true" and args.prompt_setting == "none":
        args.prompt_setting = "segment_objects"
        print("SEGMENT_OBJECTS=true: using prompt-setting=segment_objects (initialize → get_better_object via ComfyUI).")
    
    # Handle resume mode: auto-detect latest run if RESUME=true but RESUME_PATH is not set
    if args.resume and not args.resume_path:
        # Find the latest output directory for the current task
        output_root = Path(args.output_dir)
        if output_root.exists():
            # Get all timestamp directories
            timestamp_dirs = sorted([d for d in output_root.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
            for ts_dir in timestamp_dirs:
                # Check if this directory has the task we're looking for
                task_dir = ts_dir / args.task
                if task_dir.exists() and (task_dir / "generator_memory.json").exists():
                    args.resume_path = str(task_dir)
                    print(f"Auto-detected latest run to resume: {args.resume_path}")
                    break
        if not args.resume_path:
            print("Warning: RESUME=true but no previous run found. Starting fresh.")
            args.resume = False
    
    # Load dataset
    print(f"Loading static scene dataset from: {args.dataset_path}")
    tasks = load_static_scene_dataset(args.dataset_path, args.task, args.init_setting, args.test_id)
    
    if not tasks:
        print("No valid static scene tasks found in dataset!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} static scene tasks")
    for task in tasks:
        print(f"  - {task['task_name']}: {task['target_image_path']}")
    
    # For resume mode, use the parent directory of the resume path
    if args.resume_path:
        # Extract parent directory from resume path (e.g., output/static_scene/20260202_141422/test -> output/static_scene/20260202_141422)
        args.output_dir = str(Path(args.resume_path).parent)
        print(f"Resume mode: using existing output directory {args.output_dir}")
    else:
        # Create output directory only for new runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args.output_dir + "/args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    
    # Run tasks
    start_time = time.time()
    
    # Run tasks
    if args.max_workers == 1:
        # Sequential execution
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\nTask {i}/{len(tasks)}")
            task_name, success, error_msg = run_static_scene_task(task_config, args)
            
            if success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                print(f"Failed: {error_msg}")
        
        print(f"\nStatic scene task execution completed:")
        print(f"  Successful: {successful_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Total: {len(tasks)}")
    else:
        # Parallel execution
        run_static_scene_tasks_parallel(tasks, args, args.max_workers)
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
