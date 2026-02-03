import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Freestyle Rendering Test Script")
    parser.add_argument("--script", help="Path to the Blender Python script (e.g., output/.../scripts/1.py)")
    parser.add_argument("--output", default=os.getenv("OUTPUT_ROOT", "output/test_freestyle"), help="Directory to save test renders")
    parser.add_argument("--engine", default=os.getenv("RENDER_ENGINE", "BLENDER_EEVEE"), choices=["BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH", "eevee", "cycles", "workbench", "outline"], help="Render engine")
    parser.add_argument("--effect", default=os.getenv("RENDER_EFFECT", "freestyle"), choices=["none", "freestyle"], help="Render effect")
    parser.add_argument("--blender", default=os.getenv("BLENDER_COMMAND", "/Applications/Blender.app/Contents/MacOS/Blender"), help="Path to Blender executable")
    
    args = parser.parse_args()
    
    # Handle aliases
    engine_map = {
        'eevee': 'BLENDER_EEVEE',
        'cycles': 'CYCLES',
        'workbench': 'BLENDER_WORKBENCH',
        'outline': 'BLENDER_WORKBENCH',
    }
    args.engine = engine_map.get(args.engine.lower(), args.engine)

    if not args.script:
        # Try to find the latest script if not provided
        output_root = Path(os.getenv("OUTPUT_ROOT", "output/static_scene"))
        if output_root.exists():
            # Find latest timestamp dir
            timestamps = sorted([d for d in output_root.iterdir() if d.is_dir()], reverse=True)
            if timestamps:
                latest_task = timestamps[0] / os.getenv("TASK", "test")
                scripts_dir = latest_task / "scripts"
                if scripts_dir.exists():
                    scripts = sorted(list(scripts_dir.glob("*.py")), key=lambda x: int(x.stem) if x.stem.isdigit() else 0, reverse=True)
                    if scripts:
                        args.script = str(scripts[0])
                        print(f"Auto-detected latest script: {args.script}")
    
    if not args.script:
        print("Error: --script path is required or no previous scripts found.")
        sys.exit(1)
    
    # Paths
    workspace_root = Path(__file__).parent.parent.parent
    generator_script = workspace_root / "data/static_scene/generator_script.py"
    script_path = Path(args.script).absolute()
    output_dir = Path(args.output).absolute()
    
    if not generator_script.exists():
        print(f"Error: generator_script.py not found at {generator_script}")
        sys.exit(1)
    if not script_path.exists():
        print(f"Error: Input script not found at {script_path}")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment variables for Freestyle
    env = os.environ.copy()
    env['RENDER_ENGINE'] = args.engine
    env['RENDER_EFFECT'] = args.effect
    
    print(f"--- Starting Freestyle Test Render ---")
    print(f"Input Script: {script_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Engine: {args.engine}")
    print(f"Effect: {args.effect}")
    print(f"--------------------------------------")
    
    # Build command
    # Usage: blender --background [flags] -- code.py [render_dir] [save_blend]
    cmd = [
        args.blender,
        "--background",
        "--factory-startup",
        "--python", str(generator_script),
        "--",
        str(script_path),
        str(output_dir),
        str(output_dir / "test_state.blend")
    ]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Save logs
        with open(output_dir / "blender_stdout.log", "w") as f:
            f.write(result.stdout)
        with open(output_dir / "blender_stderr.log", "w") as f:
            f.write(result.stderr)
            
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"\n[SUCCESS] Test render completed. Check results in: {output_dir}")
        else:
            print(f"\n[FAILED] Blender exited with code {result.returncode}")
            
    except Exception as e:
        print(f"Error running Blender: {e}")

if __name__ == "__main__":
    main()
