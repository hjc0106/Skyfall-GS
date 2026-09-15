# single-scale training and multi-scale testing setting proposed in mip-splatting
import os
from pathlib import Path
import shlex
import subprocess
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# Interpreters and weights for the isolated Stage 2 synthesize/refine environments.
SKYFALL_PYTHON = os.environ.get("SKYFALL_PYTHON", "python")
VLM_PYTHON = os.environ.get("VLM_PYTHON", str(Path.home() / "miniconda3/envs/fixanything/bin/python3.10"))
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "./weights/Qwen3-VL-4B-Instruct")
DLORAL_PYTHON = os.environ.get("DLORAL_PYTHON", str(Path.home() / "miniconda3/envs/dloral/bin/python"))
DLORAL_WEIGHT_ROOT = os.environ.get("DLORAL_WEIGHT_ROOT", "./weights/dloral")

scenes = ["JAX_004", "JAX_068", "JAX_214", "JAX_260"]

factors = [1] * len(scenes)


dataset_dir = "./data/datasets_JAX"
output_dir = "./outputs/JAX_idu"
pre_iterative_dir = "./outputs/JAX"


dry_run = False
train = True
fused_only = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    # Base command with environment variables and Python script. The isolated
    # VLM/DLoRAL environments are selected by env vars, so the in-process
    # defaults in arguments/__init__.py pick them up.
    base_cmd = (
        f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
        f"VLM_PYTHON={shlex.quote(VLM_PYTHON)} VLM_MODEL_PATH={shlex.quote(VLM_MODEL_PATH)} "
        f"DLORAL_PYTHON={shlex.quote(DLORAL_PYTHON)} DLORAL_WEIGHT_ROOT={shlex.quote(DLORAL_WEIGHT_ROOT)} "
        f"{shlex.quote(SKYFALL_PYTHON)} train.py"
    )
    
    # Define arguments as a list for easy commenting and modification
    args = [
        f"-s {dataset_dir}/{scene}/",
        f"-m {output_dir}/{scene}",
        f"--start_checkpoint ./{pre_iterative_dir}/{scene}/chkpnt30000.pth",
        "--iterative_datasets_update",
        "--eval",
        f"--port {6209+int(gpu)}",
        "--kernel_size 0.1",
        "--resolution 1",
        "--sh_degree 1",
        "--appearance_enabled",
        "--lambda_depth 0.0",
        "--lambda_opacity 0.0",
        "--opacity_reset_interval 10000000",
        "--idu_opacity_reset_interval 5000",
        "--idu_num_samples_per_view 2",
        "--densify_grad_threshold 0.0002",
        "--datasets_type jax_v1",
        "--idu_num_cams 6",
        "--idu_grid_size 3",
        "--idu_grid_width 512",
        "--idu_grid_height 512",
        "--idu_episode_iterations 10000",
        "--idu_opacity_cooling_iterations 500",
        "--lambda_pseudo_depth 0.5",
        "--idu_densify_until_iter 9000",
        "--idu_train_ratio 0.75",
    ]
    
    # Combine base command with all arguments
    cmd = base_cmd + " " + " ".join(args)
    
    # Create log file path in the output directory
    log_file = f"{output_dir}/{scene}/log.txt"
    
    # Add logging with tee to display on terminal and save to file simultaneously
    cmd_with_tee = f"{cmd} 2>&1 | tee {log_file}"
    
    print(cmd)
    print(f"Logging output to terminal and: {log_file}")
    if not dry_run and train and not fused_only:
        # Create output directory if it doesn't exist
        os.makedirs(f"{output_dir}/{scene}", exist_ok=True)
        subprocess.run(["bash", "-o", "pipefail", "-c", cmd_with_tee], check=True)


    # Create fused ply command
    checkpoints = sorted(
        Path(output_dir, scene).glob("chkpnt*.pth"),
        key=lambda path: int(path.stem.removeprefix("chkpnt")),
    )
    if not dry_run and not checkpoints:
        raise FileNotFoundError(f"No Stage2 checkpoint in {output_dir}/{scene}")
    iteration = int(checkpoints[-1].stem.removeprefix("chkpnt")) if checkpoints else 80000
    cmd = [
        SKYFALL_PYTHON, "create_fused_ply.py", "-m", f"{output_dir}/{scene}",
        "--output_ply", f"fused/{scene}_gaussianzoom_iter_{iteration}.ply",
        "--load_from_checkpoints", "--iteration", str(iteration),
    ]
    print(shlex.join(cmd))
    if not dry_run:
        Path("fused").mkdir(exist_ok=True)
        subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)
    
    return True
        
    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set([])  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            future.result()
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

