import json
import os
import subprocess
import tempfile
from pathlib import Path


# Create temporary configuration file
def create_test_config():
	config = {
		"base_command": "python -c \"import torch; import time; print(f'Running on GPU {torch.cuda.current_device()}'); time.sleep(5)\"",
		"output_dir": "test_results",
		"param_grid": {"learning_rate": [0.001, 0.01], "batch_size": [32, 64], "layers": [1, 2]},
	}

	with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
		json.dump(config, f, indent=2)
		config_path = f.name

	return config_path


# Function to run paracuda with different parameters
def test_paracuda(gpus=2, dry_run=True):
	config_path = create_test_config()

	# Create control directory
	control_dir = Path("test_control_dir")
	control_dir.mkdir(exist_ok=True)

	# Build command
	cmd = [
		"python",
		"-m",
		"src.paracuda.paracuda_run",
		"--config",
		config_path,
		"--gpus",
		str(gpus),
		"--control_dir",
		str(control_dir),
		"--log-level",
		"INFO",
	]

	if dry_run:
		cmd.append("--dry-run")

	print(f"Running command: {' '.join(cmd)}")
	subprocess.run(cmd)

	# Clean up
	os.unlink(config_path)


# Test with different GPU configurations
print("Testing with 2 GPUs (dry run):")
test_paracuda(gpus=2, dry_run=True)

print("\nTesting with 4 GPUs (dry run):")
test_paracuda(gpus=4, dry_run=True)

# For actual runs (be careful with this as it will execute tasks)
# Uncomment the following lines to test without dry run
# print("\nTesting with 2 GPUs (actual run):")
# test_paracuda(gpus=2, dry_run=False)


# Helper function to analyze GPU assignment
def analyze_gpu_assignments(gpus=4):
	"""Simulate and print GPU assignments to verify distribution"""
	print(f"\nAnalyzing GPU assignments with {gpus} GPUs:")

	# Create a test parameter grid
	param_grid = {"learning_rate": [0.001, 0.01], "batch_size": [32, 64], "layers": [1, 2]}

	# Generate combinations (simplified version of ParameterGrid)
	combinations = []
	from itertools import product

	keys = param_grid.keys()
	values = param_grid.values()

	for combo in product(*values):
		combinations.append(dict(zip(keys, combo)))

	# Print GPU assignment for each task
	gpu_counts = {i: 0 for i in range(gpus)}

	print("Task | Parameters | GPU")
	print("-" * 50)

	for idx, params in enumerate(combinations):
		gpu_id = idx % gpus
		gpu_counts[gpu_id] += 1
		param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
		print(f"{idx:4d} | {param_str:30s} | GPU {gpu_id}")

	print("\nGPU utilization:")
	for gpu_id, count in gpu_counts.items():
		print(f"GPU {gpu_id}: {count} tasks")


# Analyze GPU assignments with different GPU counts
analyze_gpu_assignments(gpus=2)
analyze_gpu_assignments(gpus=4)
