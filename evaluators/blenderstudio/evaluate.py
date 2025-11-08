import argparse
import os
import argparse
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate BlenderStudio results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    args = parser.parse_args()
    
    ref_based_eval_path = 'evaluators/blenderstudio/ref_based_eval.py'
    ref_free_eval_path = 'evaluators/blenderstudio/ref_free_eval.py'
    
    # run two evaluations in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(subprocess.run, [
            "python", ref_based_eval_path,
            args.test_id
        ], check=True)
        future2 = executor.submit(subprocess.run, [
            "python", ref_free_eval_path,
            args.test_id
        ], check=True)
        for future in as_completed([future1, future2]):
            future.result()