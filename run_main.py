#!/usr/bin/env python3
"""
Wrapper script to ensure multiprocessing start method is set to 'spawn'
before any imports that might initialize CUDA or multiprocessing.
This is required for vLLM to work correctly with CUDA.
"""
import os
import sys
import multiprocessing

# CRITICAL: Set multiprocessing start method BEFORE any other imports
# This must happen before torch, vllm, or any CUDA-related imports
os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"

# Set the start method programmatically
try:
    multiprocessing.set_start_method("spawn", force=True)
    print("✓ Multiprocessing start method set to 'spawn'", file=sys.stderr)
except RuntimeError as e:
    # If start method is already set, check if it's spawn
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        print(
            f"ERROR: Multiprocessing start method is '{current_method}', not 'spawn'.",
            file=sys.stderr,
        )
        print(
            "Cannot change start method after it's been set. "
            "Please ensure PYTHON_MULTIPROCESSING_START_METHOD=spawn is set "
            "before starting Python.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"✓ Multiprocessing start method already set to 'spawn'", file=sys.stderr)

# Now import and run the main module
if __name__ == "__main__":
    from EncoderSAE.main import main
    import fire

    fire.Fire(main)
