#!/usr/bin/env python3
"""
BlenderGym Baseline Runner for AgenticVerifier
Loads completed BlenderGym tasks from output directory and runs tournament-style evaluation
using VLM to compare images and select winners.
"""
import os
import sys
import json
import time
import argparse
import base64
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import API keys from runners directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_keys import OPENAI_API_KEY
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import openai
from tqdm import tqdm

# Global CLIP model/processor to share across threads
GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def ensure_clip_loaded():
    """
    Lazily load the global CLIP model and processor once per process.
    """
    global GLOBAL_CLIP_MODEL, GLOBAL_CLIP_PROCESSOR
    if GLOBAL_CLIP_MODEL is None or GLOBAL_CLIP_PROCESSOR is None:
        GLOBAL_CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        GLOBAL_CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clip_similarity(image1, image2):
    """
    Compute the CLIP similarity between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The CLIP similarity between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Ensure global model is initialized
    ensure_clip_loaded()

    # Preprocess the images
    images = [image1, image2]
    inputs = GLOBAL_CLIP_PROCESSOR(images=images, return_tensors="pt")

    # Compute the features for the images
    with torch.no_grad():
        features = GLOBAL_CLIP_MODEL.get_image_features(**inputs)

    # Compute the cosine similarity between the image features
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=-1)

    return sim.item()


def photometric_loss(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute the photometric loss between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The photometric loss between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert images to numpy arrays
    img1_array = np.array(image1)[:, :, :3]
    img2_array = np.array(image2)[:, :, :3]

    # Normalize images to [0, 1]
    img1_norm = img1_array.astype(np.float32) / 255.0
    img2_norm = img2_array.astype(np.float32) / 255.0

    # Compute the squared difference between the normalized images
    diff = np.square(img1_norm - img2_norm)

    # Compute the mean squared error
    mse = np.mean(diff)
    return mse


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string for OpenAI API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_compare_images(image1_path: str, image2_path: str, target_path: str, 
                      api_key: str, base_url: str, model: str = "gpt-4o") -> int:
    """
    Use VLM to compare two images and determine which is closer to target.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image  
        target_path: Path to target image
        api_key: OpenAI API key
        base_url: OpenAI base URL
        model: Vision model to use
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        # Encode images
        image1_b64 = encode_image(image1_path)
        image2_b64 = encode_image(image2_path)
        target_b64 = encode_image(target_path)
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at comparing 3D rendered images. I will show you two rendered images and a target image. Please determine which of the two rendered images is closer to the target image in terms of visual similarity, lighting, materials, geometry, and overall appearance. Respond with only '1' if the first image is closer to the target, or '2' if the second image is closer to the target."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": "Target image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        
        # Parse response
        result = response.choices[0].message.content.strip()
        if result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            # Fallback: use CLIP similarity
            print(f"Unexpected VLM response: {result}, using CLIP fallback")
            return clip_fallback_comparison(image1_path, image2_path, target_path)
            
    except Exception as e:
        print(f"VLM comparison failed: {e}, using CLIP fallback")
        return clip_fallback_comparison(image1_path, image2_path, target_path)


def clip_fallback_comparison(image1_path: str, image2_path: str, target_path: str) -> int:
    """
    Fallback comparison using CLIP similarity when VLM fails.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        target_path: Path to target image
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        target = Image.open(target_path)
        
        sim1 = clip_similarity(image1, target)
        sim2 = clip_similarity(image2, target)
        
        return 1 if sim1 > sim2 else 2
    except Exception as e:
        print(f"CLIP fallback also failed: {e}, defaulting to image1")
        return 1


def load_blenderstudio_dataset(base_path: str, task_name: str, task_id: Optional[str] = None) -> List[Dict]:
    """
    Load BlenderStudio dataset structure (similar to load_blendergym_dataset in blenderstudio.py).
    
    Args:
        base_path: Path to BlenderStudio dataset root
        task_name: Task name (e.g., 'level1', 'level2', 'level3', or 'all')
        task_id: Optional task ID to filter specific tasks
        
    Returns:
        List of task configurations
    """
    tasks = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: BlenderStudio dataset path does not exist: {base_path}")
        return tasks
    
    if task_name == 'all':
        task_list = ['level1', 'level2', 'level3']
    else:
        task_list = [task_name]
    
    # If task_id is not None, only run the task_id
    if task_id is not None:
        task_dirs = [(base_path / f"{task_name}/{task_id}", task_name + '-' + task_id[-1])]
    # Otherwise, run all tasks in the task_list
    else:
        task_dirs = []
        for task in task_list:
            task_path = base_path / task
            for task_dir in task_path.glob("*"):
                task_name_final = task + '-' + str(task_dir.name.split('/')[-1][-1])
                task_dirs.append((task_dir, task_name_final))
    
    for task_dir, task_name_final in task_dirs:
        # Check for required files
        start_code_path = task_dir / "start.py"
        start_renders_dir = task_dir / "renders" / "start"
        goal_renders_dir = task_dir / "renders" / "goal"
        blender_file = task_dir / "blender_file.blend"
        
        if not start_code_path.exists():
            print(f"Warning: start.py not found in {task_dir}")
            continue
        
        if not goal_renders_dir.exists() or not start_renders_dir.exists():
            print(f"Warning: renders directory not found: {goal_renders_dir}")
            continue
        
        if not blender_file.exists():
            print(f"Warning: blender_file.blend not found in {task_dir}")
            continue
        
        task_name_path = task_dir / "task.txt"
        if os.path.exists(task_name_path):
            task_description = open(task_name_path, 'r').read()
        else:
            task_description = ""
        
        task_config = {
            "task_name": task_name_final,
            "task_dir": str(task_dir),
            "init_code_path": str(start_code_path),
            "init_image_path": str(start_renders_dir),
            "target_image_path": str(goal_renders_dir),
            "blender_file": str(blender_file),
            "target_description": task_description,
        }
        tasks.append(task_config)
        print(f"Found task: {task_name_final}")
    
    return tasks


def execute_blender_code(blender_command: str, blender_file: str, blender_script: str, 
                         code: str, script_save_dir: Path, render_save_dir: Path, 
                         count: int, gpu_devices: Optional[str] = None) -> Tuple[bool, str, Dict]:
    """
    Execute Blender Python code and render images.
    
    Args:
        blender_command: Path to Blender executable
        blender_file: Path to Blender file
        blender_script: Path to Blender execution script
        code: Python code to execute
        script_save_dir: Directory to save the code file
        render_save_dir: Directory to save rendered images
        count: Counter for file naming
        gpu_devices: GPU devices string (e.g., "0,1")
        
    Returns:
        Tuple of (success: bool, error_message: str, result_dict: Dict)
        result_dict contains rendered image paths if successful
    """
    code_file = script_save_dir / f"{count}.py"
    render_file = render_save_dir / f"{count}"
    
    # Save code to file
    script_save_dir.mkdir(parents=True, exist_ok=True)
    with open(code_file, "w") as f:
        f.write(code)
    
    # Create render directory
    render_file.mkdir(parents=True, exist_ok=True)
    for img in os.listdir(render_file):
        os.remove(os.path.join(render_file, img))
    
    # Execute Blender
    cmd = [
        blender_command,
        "--background", blender_file,
        "--python", blender_script,
        "--", str(code_file), str(render_file)
    ]
    
    if gpu_devices:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    else:
        env = None
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        if result.returncode != 0:
            return False, result.stderr + result.stdout, {}
        
        # Check if render files exist
        render_files = list(render_file.glob("*.png"))
        if len(render_files) == 0:
            return False, "No render output generated", {}
        
        # Return render paths
        render_paths = {
            f"render{i+1}": str(rf) for i, rf in enumerate(sorted(render_files))
        }
        return True, "", render_paths
        
    except subprocess.TimeoutExpired:
        return False, "Execution timeout", {}
    except Exception as e:
        return False, str(e), {}


def generate_candidate_codes(start_image_path: str, current_image_path: str, current_code: str,
                             target_image_path: str, task_description: str,
                             api_key: str, base_url: str, model: str = "gpt-4o",
                             num_candidates: int = 4) -> List[str]:
    """
    Use GPT to generate multiple candidate codes to transform current image to target.
    
    Args:
        start_image_path: Path to starting image
        current_image_path: Path to current image
        current_code: Current Blender Python code
        target_image_path: Path to target image
        task_description: Task description text
        api_key: OpenAI API key
        base_url: OpenAI base URL
        model: Model name
        num_candidates: Number of candidate codes to generate (3-4)
        
    Returns:
        List of candidate code strings
    """
    try:
        # Encode images
        start_b64 = encode_image(start_image_path)
        current_b64 = encode_image(current_image_path)
        target_b64 = encode_image(target_image_path)
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing Blender Python code to transform 3D scenes. Given a starting image, current image, current code, and target image, generate multiple candidate code solutions."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Task description: {task_description}

You are given:
1. Starting image (initial state)
2. Current image (current state after applying current code)
3. Current Blender Python code
4. Target image (desired final state)

Please generate {num_candidates} different candidate Blender Python code solutions that can transform the current image closer to the target image. Each candidate should be a complete, runnable Blender Python script.

Current code:
```python
{current_code}
```

Please output {num_candidates} complete code solutions, separated by "===CANDIDATE_1===", "===CANDIDATE_2===", etc. Each code block should be complete and executable."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{start_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Starting image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Current image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Target image:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        # Parse response to extract candidate codes
        content = response.choices[0].message.content
        candidates = []
        
        # Split by candidate markers
        parts = content.split("===CANDIDATE_")
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            # Extract code between markers
            if "===" in part:
                code = part.split("===", 1)[1]
                # Remove markdown code blocks if present
                code = code.replace("```python", "").replace("```", "").strip()
                candidates.append(code)
            else:
                # Last candidate or no marker
                code = part.strip()
                code = code.replace("```python", "").replace("```", "").strip()
                if code:
                    candidates.append(code)
        
        # If no markers found, try to extract code blocks
        if len(candidates) == 0:
            import re
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', content, re.DOTALL)
            candidates = code_blocks[:num_candidates]
        
        # Ensure we have the right number of candidates
        while len(candidates) < num_candidates and len(candidates) > 0:
            candidates.append(candidates[-1])  # Duplicate last candidate if needed
        
        return candidates[:num_candidates]
        
    except Exception as e:
        print(f"Error generating candidate codes: {e}")
        # Return empty codes as fallback
        return [current_code] * num_candidates


def tournament_select_best(candidate_results: List[Dict], target_image_path: str,
                           api_key: str, base_url: str, model: str = "gpt-4o") -> int:
    """
    Run tournament to select the best candidate using VLM comparison.
    
    Args:
        candidate_results: List of dicts with keys 'render_path' (path to render1.png)
        target_image_path: Path to target image
        api_key: OpenAI API key
        base_url: OpenAI base URL
        model: Vision model name
        
    Returns:
        Index of the winning candidate
    """
    if len(candidate_results) == 0:
        return 0
    
    if len(candidate_results) == 1:
        return 0
    
    # Tournament: keep pairing and comparing until one winner
    current_candidates = list(range(len(candidate_results)))
    
    while len(current_candidates) > 1:
        next_round = []
        
        # Pair up candidates
        for i in range(0, len(current_candidates), 2):
            if i + 1 < len(current_candidates):
                idx1 = current_candidates[i]
                idx2 = current_candidates[i + 1]
                
                img1_path = candidate_results[idx1]['render_path']
                img2_path = candidate_results[idx2]['render_path']
                
                # Compare which is closer to target
                winner = vlm_compare_images(
                    img1_path, img2_path, target_image_path,
                    api_key, base_url, model
                )
                
                # Winner is 1 or 2, convert to index
                winner_idx = idx1 if winner == 1 else idx2
                next_round.append(winner_idx)
            else:
                # Odd number, last one gets bye
                next_round.append(current_candidates[i])
        
        current_candidates = next_round
    
    return current_candidates[0]


def run_iterative_alchemy(task_config: Dict, args) -> Dict:
    """
    Run iterative alchemy process: generate candidates -> tournament -> update -> repeat.
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
        
    Returns:
        Dictionary with results
    """
    
    task_name = task_config['task_name']
    print(f"\n{'='*60}")
    print(f"Running iterative alchemy for task: {task_name}")
    print(f"{'='*60}")
    
    # Setup paths
    output_dir = Path(args.output_dir) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_save_dir = output_dir / "scripts"
    render_save_dir = output_dir / "renders"
    
    # Get initial code
    with open(task_config['init_code_path'], 'r') as f:
        current_code = f.read()
    
    # Get image paths
    start_image_dir = Path(task_config['init_image_path'])
    target_image_dir = Path(task_config['target_image_path'])
    
    # Find render1.png files
    start_images = list(start_image_dir.glob("render*.png"))
    target_images = list(target_image_dir.glob("render*.png"))
    
    if not start_images or not target_images:
        return {
            "task_name": task_name,
            "error": "Missing start or target images",
            "success": False
        }
    
    start_image_path = str(start_images[0])  # Use first render
    target_image_path = str(target_images[0])  # Use first target
    
    # Initialize current image path (starts from initial image)
    current_image_path = start_image_path
    
    # Copy blender file to output
    shutil.copy(task_config['blender_file'], output_dir / "blender_file.blend")
    blender_file = str(output_dir / "blender_file.blend")
    
    iteration_results = []
    
    # Iterative process
    for iteration in range(args.max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.max_iterations} ---")
        
        # Generate candidate codes
        print("Generating candidate codes...")
        candidate_codes = generate_candidate_codes(
            start_image_path=start_image_path,
            current_image_path=current_image_path,
            current_code=current_code,
            target_image_path=target_image_path,
            task_description=task_config.get('target_description', ''),
            api_key=args.api_key,
            base_url=args.openai_base_url,
            model=args.vision_model,
            num_candidates=args.num_candidates
        )
        
        print(f"Generated {len(candidate_codes)} candidate codes")
        
        # Execute all candidate codes
        candidate_results = []
        for i, code in enumerate(candidate_codes):
            print(f"  Executing candidate {i+1}/{len(candidate_codes)}...")
            success, error_msg, render_paths = execute_blender_code(
                blender_command=args.blender_command,
                blender_file=blender_file,
                blender_script=args.blender_script,
                code=code,
                script_save_dir=script_save_dir,
                render_save_dir=render_save_dir,
                count=iteration * args.num_candidates + i + 1,
                gpu_devices=args.gpu_devices
            )
            
            if success and render_paths:
                # Find render1.png
                render_path = render_paths.get('render1', list(render_paths.values())[0])
                candidate_results.append({
                    'code': code,
                    'render_path': render_path,
                    'index': i
                })
                print(f"    ✓ Candidate {i+1} executed successfully")
            else:
                print(f"    ✗ Candidate {i+1} failed: {error_msg}")
        
        if len(candidate_results) == 0:
            print("No candidates executed successfully, stopping.")
            break
        
        # Run tournament to select best
        print(f"Running tournament with {len(candidate_results)} candidates...")
        winner_idx = tournament_select_best(
            candidate_results=candidate_results,
            target_image_path=target_image_path,
            api_key=args.api_key,
            base_url=args.openai_base_url,
            model=args.vision_model
        )
        
        winner = candidate_results[winner_idx]
        print(f"  Winner: Candidate {winner_idx + 1}")
        
        # Update current state
        current_code = winner['code']
        current_image_path = winner['render_path']
        
        # Calculate metrics for winner
        winner_image = Image.open(current_image_path)
        target_image = Image.open(target_image_path)
        
        clip_score = clip_similarity(winner_image, target_image)
        pl_score = photometric_loss(winner_image, target_image)
        
        iteration_results.append({
            'iteration': iteration + 1,
            'winner_candidate': winner_idx + 1,
            'n_clip': 1 - clip_score,
            'pl': pl_score,
            'winner_code': current_code
        })
        
        print(f"  Metrics: n_clip={1-clip_score:.4f}, pl={pl_score:.4f}")
        
        # Check if we've reached target (optional early stopping)
        if clip_score > args.target_similarity_threshold:
            print(f"  Target similarity reached ({clip_score:.4f} > {args.target_similarity_threshold})")
            break
    
    # Final metrics
    final_image = Image.open(current_image_path)
    target_image = Image.open(target_image_path)
    final_clip = 1 - clip_similarity(final_image, target_image)
    final_pl = photometric_loss(final_image, target_image)
    
    return {
        "task_name": task_name,
        "success": True,
        "iterations": iteration_results,
        "final_metrics": {
            "n_clip": final_clip,
            "pl": final_pl
        },
        "final_code": current_code,
        "final_image_path": current_image_path
    }


def get_max_rounds_for_task(task_name: str) -> int:
    """
    Get the maximum number of rounds for a task from the reference directory.
    
    Args:
        task_name: Name of the task (e.g., 'blendshape1')
        
    Returns:
        Maximum round number found in the reference directory
    """
    reference_dir = Path(f"output/blendergym/gpt-4o/{task_name}/renders")
    if not reference_dir.exists():
        print(f"Warning: Reference directory not found: {reference_dir}")
        return 8  # Default fallback
    
    max_round = 0
    for round_dir in reference_dir.iterdir():
        if round_dir.is_dir() and round_dir.name.isdigit():
            round_num = int(round_dir.name)
            max_round = max(max_round, round_num)
    
    return max_round


def load_tasks_from_output(output_dir: str) -> List[Dict]:
    """
    Load completed tasks from output directory.
    
    Args:
        output_dir: Path to output directory containing task results
        
    Returns:
        List of task configurations
    """
    tasks = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_path}")
        return tasks
    
    # Find all task directories (e.g., blendshape1, geometry5, etc.)
    for task_dir in output_path.iterdir():
        if task_dir.is_dir() and task_dir.name != "_evaluation":
            renders_dir = task_dir / "renders"
            if renders_dir.exists():
                task_name = task_dir.name
                
                # Get maximum rounds from reference directory
                max_rounds = get_max_rounds_for_task(task_name)
                print(f"Task {task_name}: using max rounds = {max_rounds}")
                
                # Check for round directories (1 to max_rounds)
                round_dirs = []
                for i in range(1, max_rounds + 1):
                    round_dir = renders_dir / str(i)
                    if round_dir.exists():
                        # Check for render files
                        render1 = round_dir / "render1.png"
                        if render1.exists():
                            round_dirs.append(i)
                
                # Find corresponding target images
                target_renders_dir = Path(f"data/blendergym/{task_name}/renders/goal")
                if target_renders_dir.exists():
                    target_render1 = target_renders_dir / "render1.png"
                    
                    if target_render1.exists():
                        task_config = {
                            "task_name": task_name,
                            "task_dir": str(task_dir),
                            "renders_dir": str(renders_dir),
                            "target_renders_dir": str(target_renders_dir),
                            "round_dirs": sorted(round_dirs),
                            "max_rounds": max_rounds
                        }
                        tasks.append(task_config)
                        
                        if len(round_dirs) >= 2:
                            print(f"Found task: {task_name} with {len(round_dirs)} rounds (max_rounds={max_rounds}) - will run tournament")
                        elif len(round_dirs) == 1:
                            print(f"Found task: {task_name} with {len(round_dirs)} round (max_rounds={max_rounds}) - will auto-win")
                        else:
                            print(f"Found task: {task_name} with {len(round_dirs)} rounds (max_rounds={max_rounds}) - will get penalty score")
                    else:
                        print(f"Warning: Target renders not found for {task_name}")
                else:
                    print(f"Warning: Target renders directory not found for {task_name}")
    
    return tasks


def run_tournament(task_config: Dict, args) -> Dict:
    """
    Run tournament-style evaluation for a single task.
    
    Args:
        task_config: Task configuration
        args: Command line arguments
        
    Returns:
        Dictionary with tournament results
    """
    task_name = task_config['task_name']
    renders_dir = Path(task_config['renders_dir'])
    target_renders_dir = Path(task_config['target_renders_dir'])
    
    print(f"\nRunning tournament for task: {task_name}")
    
    # Get all available round directories
    available_rounds = task_config['round_dirs']
    
    # Prepare images for tournament (use rounds 1 to max_rounds)
    max_rounds = task_config.get('max_rounds', 8)  # Default fallback
    images = []
    for round_num in range(1, max_rounds + 1):
        if round_num in available_rounds:  # Only use rounds that exist
            round_dir = renders_dir / str(round_num)
            render1_path = round_dir / "render1.png"
            render2_path = round_dir / "render2.png"
            
            if render1_path.exists() and render2_path.exists():
                images.append({
                    'round': round_num,
                    'render1': str(render1_path),
                    'render2': str(render2_path)
                })
            elif render1_path.exists():
                images.append({
                    'round': round_num,
                    'render1': str(render1_path),
                    'render2': str(render1_path)
                })
    
    # Handle special cases: 0 rounds or 1 round
    if len(images) == 0:
        print(f"Warning: No rounds available for {task_name}, assigning penalty score")
        return {
            "task_name": task_name,
            "max_rounds": max_rounds,
            "total_participants": 0,
            "special_case": "no_rounds",
            "final_metrics": {
                "n_clip_render1": 1.0,  # Maximum penalty
                "n_clip_render2": 1.0,  # Maximum penalty
                "avg_n_clip": 1.0,      # Maximum penalty
                "pl_render1": 1.0,      # Maximum penalty
                "pl_render2": 1.0,      # Maximum penalty
                "avg_pl": 1.0           # Maximum penalty
            }
        }
    elif len(images) == 1:
        print(f"Only 1 round available for {task_name}, auto-winning")
        # Calculate metrics for the single round
        single_image = images[0]
        target_render1 = str(target_renders_dir / "render1.png")
        target_render2 = str(target_renders_dir / "render2.png")
        
        if not os.path.exists(target_render2):
            target_render2 = target_render1
            
        # Load images for metric calculation
        winner_render1 = Image.open(single_image['render1'])
        winner_render2 = Image.open(single_image['render2'])
        target_img1 = Image.open(target_render1)
        target_img2 = Image.open(target_render2)
        
        # CLIP metrics (1 - similarity = distance)
        clip1 = 1 - clip_similarity(winner_render1, target_img1)
        clip2 = 1 - clip_similarity(winner_render2, target_img2)
        avg_clip = (clip1 + clip2) / 2
        
        # Photometric loss
        pl1 = photometric_loss(winner_render1, target_img1)
        pl2 = photometric_loss(winner_render2, target_img2)
        avg_pl = (pl1 + pl2) / 2
        
        return {
            "task_name": task_name,
            "max_rounds": max_rounds,
            "total_participants": 1,
            "special_case": "auto_win",
            "final_winner": single_image,
            "final_metrics": {
                "n_clip_render1": clip1,
                "n_clip_render2": clip2,
                "avg_n_clip": avg_clip,
                "pl_render1": pl1,
                "pl_render2": pl2,
                "avg_pl": avg_pl
            }
        }
    
    # Target images
    target_render1 = str(target_renders_dir / "render1.png")
    target_render2 = str(target_renders_dir / "render2.png")
    
    if not os.path.exists(target_render2):
        target_render2 = target_render1
    
    # Tournament: dynamic rounds with bye logic
    current_images = images.copy()
    
    tournament_results = {
        "task_name": task_name,
        "max_rounds": max_rounds,
        "total_participants": len(images),
        "rounds": []
    }
    
    round_num = 1
    while len(current_images) > 1:
        print(f"  Tournament Round {round_num}: {len(current_images)} images remaining")
        
        round_results = {
            "round": round_num,
            "participants": len(current_images),
            "comparisons": [],
            "byes": []
        }
        
        # Pair up images for comparison
        next_round_images = []
        
        # Handle byes: if odd number of participants, the last one gets a bye
        if len(current_images) % 2 == 1:
            bye_image = current_images[-1]
            next_round_images.append(bye_image)
            round_results["byes"].append(bye_image['round'])
            print(f"    Round {bye_image['round']} gets a bye")
            current_images = current_images[:-1]  # Remove the bye image from current round
        
        # Pair up remaining images for comparison
        for i in range(0, len(current_images), 2):
            if i + 1 < len(current_images):
                img1 = current_images[i]
                img2 = current_images[i + 1]
                
                # Compare only render1 vs render1 to determine tournament winner
                winner_render1 = vlm_compare_images(
                    img1['render1'], img2['render1'], target_render1,
                    args.api_key, args.openai_base_url, args.vision_model
                )
                
                # Winner is determined solely by render1 comparison
                winner_idx = winner_render1 - 1  # Convert to 0-based index
                
                winner = current_images[i + winner_idx]
                next_round_images.append(winner)
                
                round_results["comparisons"].append({
                    "img1_round": img1['round'],
                    "img2_round": img2['round'],
                    "winner_render1": winner_render1,
                    "final_winner": img1['round'] if winner_idx == 0 else img2['round']
                })
        
        tournament_results["rounds"].append(round_results)
        current_images = next_round_images
        round_num += 1
    
    # Final winner
    if current_images:
        final_winner = current_images[0]
        tournament_results["final_winner"] = final_winner
        
        # Calculate final metrics
        winner_render1 = Image.open(final_winner['render1'])
        winner_render2 = Image.open(final_winner['render2'])
        target_img1 = Image.open(target_render1)
        target_img2 = Image.open(target_render2)
        
        # CLIP metrics (1 - similarity = distance)
        clip1 = 1 - clip_similarity(winner_render1, target_img1)
        clip2 = 1 - clip_similarity(winner_render2, target_img2)
        avg_clip = (clip1 + clip2) / 2
        
        # Photometric loss
        pl1 = photometric_loss(winner_render1, target_img1)
        pl2 = photometric_loss(winner_render2, target_img2)
        avg_pl = (pl1 + pl2) / 2
        
        tournament_results["final_metrics"] = {
            "n_clip_render1": clip1,
            "n_clip_render2": clip2,
            "avg_n_clip": avg_clip,
            "pl_render1": pl1,
            "pl_render2": pl2,
            "avg_pl": avg_pl
        }
        
        print(f"  Final winner: Round {final_winner['round']}")
        print(f"  Final metrics: n_clip={avg_clip:.4f}, pl={avg_pl:.4f}")
    
    return tournament_results


def run_tasks_parallel(tasks: List[Dict], args, max_workers: int = 4) -> Dict:
    """
    Run tournament evaluations in parallel.
    
    Args:
        tasks: List of task configurations
        args: Command line arguments
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with all results
    """
    results = {
        "tasks": [],
        "summary": {
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "execution_time": 0
        }
    }
    
    print(f"\nStarting parallel tournament evaluation with max {max_workers} workers...")
    print(f"Total tasks: {len(tasks)}")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_tournament, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Running tournaments"):
            task_config = future_to_task[future]
            try:
                result = future.result()
                results["tasks"].append(result)
                
                if "error" not in result:
                    results["summary"]["successful_tasks"] += 1
                    print(f"✅ {result['task_name']} completed successfully")
                else:
                    results["summary"]["failed_tasks"] += 1
                    print(f"❌ {result['task_name']} failed: {result['error']}")
                    
            except Exception as e:
                results["summary"]["failed_tasks"] += 1
                error_result = {
                    "task_name": task_config['task_name'],
                    "error": str(e)
                }
                results["tasks"].append(error_result)
                print(f"❌ {task_config['task_name']} failed with exception: {e}")
    
    end_time = time.time()
    results["summary"]["execution_time"] = end_time - start_time
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BlenderGym Baseline Tournament Runner / Iterative Alchemy")
    
    # Mode selection
    parser.add_argument("--mode", choices=['tournament', 'iterative'], default='tournament',
                       help="Mode: 'tournament' for evaluating existing results, 'iterative' for iterative generation")
    
    # Input parameters (for tournament mode)
    parser.add_argument("--test-id", default=None, help="Test ID (e.g., gpt-4o-bestofn) - required for tournament mode")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    
    # Input parameters (for iterative mode)
    parser.add_argument("--dataset-path", default="data/blenderstudio", help="Path to BlenderStudio dataset (for iterative mode)")
    parser.add_argument("--task", choices=['all', 'level1', 'level2', 'level3'], default='all', help="Task name (for iterative mode)")
    parser.add_argument("--task-id", default=None, help="Specific task ID to run (for iterative mode)")
    
    # Iterative alchemy parameters
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations (for iterative mode)")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of candidate codes to generate per iteration")
    parser.add_argument("--target-similarity-threshold", type=float, default=0.95, help="CLIP similarity threshold for early stopping")
    
    # Blender parameters (for iterative mode)
    parser.add_argument("--blender-command", default="utils/Infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blenderstudio/generator_script.py", help="Blender execution script")
    parser.add_argument("--gpu-devices", default=None, help="GPU devices string (e.g., '0,1')")
    
    # VLM parameters
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=OPENAI_API_KEY, help="OpenAI API key")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--task-filter", help="Filter tasks by name pattern (e.g., 'blendshape')")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Set GPU devices if not provided
    if args.gpu_devices is None:
        args.gpu_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    
        if not args.output_dir:
            args.output_dir = f"output/alchemy/{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Load dataset
        print(f"Loading BlenderStudio dataset from: {args.dataset_path}")
        tasks = load_blenderstudio_dataset(args.dataset_path, args.task, args.task_id)
        
        if not tasks:
            print("No valid tasks found!")
            sys.exit(1)
        
        # Filter tasks if specified
        if args.task_filter:
            tasks = [t for t in tasks if args.task_filter in t["task_name"]]
            print(f"Filtered to {len(tasks)} tasks matching '{args.task_filter}'")
        
        if not tasks:
            print("No tasks match the specified filters!")
            sys.exit(1)
        
        print(f"Found {len(tasks)} tasks for iterative alchemy")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args to json
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(convert_numpy_types(args.__dict__), f, indent=2)
        
        # Save task list
        with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
            json.dump(tasks, f, indent=2)
        
        # Ensure CLIP is loaded
        ensure_clip_loaded()
        
        # Run iterative alchemy for each task
        print(f"\nStarting iterative alchemy evaluation with max {args.max_workers} workers...")
        print(f"Total tasks: {len(tasks)}")
        
        start_time = time.time()
        results = {
            "tasks": [],
            "summary": {
                "total_tasks": len(tasks),
                "successful_tasks": 0,
                "failed_tasks": 0,
                "execution_time": 0
            }
        }
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_iterative_alchemy, task_config, args): task_config 
                for task_config in tasks
            }
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Running alchemy"):
                task_config = future_to_task[future]
                try:
                    result = future.result()
                    results["tasks"].append(result)
                    
                    if result.get("success", False):
                        results["summary"]["successful_tasks"] += 1
                        print(f"✅ {result['task_name']} completed successfully")
                    else:
                        results["summary"]["failed_tasks"] += 1
                        print(f"❌ {result['task_name']} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    results["summary"]["failed_tasks"] += 1
                    error_result = {
                        "task_name": task_config['task_name'],
                        "error": str(e),
                        "success": False
                    }
                    results["tasks"].append(error_result)
                    print(f"❌ {task_config['task_name']} failed with exception: {e}")
        
        end_time = time.time()
        results["summary"]["execution_time"] = end_time - start_time
        
        # Save results
        results_path = os.path.join(args.output_dir, "alchemy_results.json")
        with open(results_path, "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ITERATIVE ALCHEMY SUMMARY")
        print(f"{'='*60}")
        print(f"Total tasks: {results['summary']['total_tasks']}")
        print(f"Successful: {results['summary']['successful_tasks']}")
        print(f"Failed: {results['summary']['failed_tasks']}")
        print(f"Execution time: {results['summary']['execution_time']:.2f} seconds")
        print(f"Results saved to: {args.output_dir}")
        
        # Calculate averages for successful tasks
        successful_tasks = [t for t in results["tasks"] if t.get("success", False)]
        if successful_tasks:
            avg_n_clip = sum(t["final_metrics"]["n_clip"] for t in successful_tasks) / len(successful_tasks)
            avg_pl = sum(t["final_metrics"]["pl"] for t in successful_tasks) / len(successful_tasks)
            print(f"\nAverage metrics across successful tasks:")
            print(f"  Average n_clip: {avg_n_clip:.4f}")
            print(f"  Average pl: {avg_pl:.4f}")
    


if __name__ == "__main__":
    main()