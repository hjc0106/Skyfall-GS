import os
import shlex
from pathlib import Path
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import subprocess
import datetime

# Interpreters and weights for the isolated Stage 2 synthesize/refine environments.
SKYFALL_PYTHON = os.environ.get("SKYFALL_PYTHON", "python")
VLM_PYTHON = os.environ.get("VLM_PYTHON", str(Path.home() / "miniconda3/envs/fixanything/bin/python3.10"))
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "./weights/Qwen3-VL-4B-Instruct")
DLORAL_PYTHON = os.environ.get("DLORAL_PYTHON", str(Path.home() / "miniconda3/envs/dloral/bin/python"))
DLORAL_WEIGHT_ROOT = os.environ.get("DLORAL_WEIGHT_ROOT", "./weights/dloral")

# Fixed scene
# SCENE = "JAX_068"

# Base paths
dataset_dir = "./data/datasets_NYC"
output_dir = "./outputs/NYC_idu"

# Execution settings
dry_run = False
train = True

def ensure_dir_exists(directory):
    """Ensure the directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_command(cmd, log_file, append=True):
    """Run a command and log the output to a file while also printing to console."""
    mode = 'a' if append else 'w'
    
    # Add timestamp to log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}] Executing: {cmd}\n"
    
    # Print to console
    print(log_entry)
    
    # Write to log file
    with open(log_file, mode) as f:
        f.write(log_entry)
    
    # Run the command and capture output in real-time
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Write output to log file and console in real-time
    with open(log_file, 'a') as f:
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)        # Write to log file
    
    # Wait for the process to complete and get return code
    return_code = process.wait()
    
    # Log completion status
    status = "SUCCESS" if return_code == 0 else f"FAILED (code: {return_code})"
    completion_entry = f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command completed: {status}\n, cmd: {cmd}\n"
    
    print(completion_entry)
    with open(log_file, 'a') as f:
        f.write(completion_entry)
    
    return return_code == 0

def run_job(gpu, model_path, command_params):
    """
    Run a job with the given parameters on a specific GPU.
    
    Args:
        gpu: GPU ID to use
        model_path: Path to the model (will be used as output directory)
        command_params: String containing all command parameters
    """
    # Ensure output directory exists
    full_output_dir = f"{output_dir}/{model_path}"
    ensure_dir_exists(full_output_dir)
    
    # Set up log file
    log_file = f"{full_output_dir}/log.txt"
    
    # Log job start info (create new log file)
    with open(log_file, 'w') as f:
        f.write(f"=== Job Started: {model_path} ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {gpu}\n")
        f.write(f"Parameters: {command_params}\n\n")

    scene = model_path
    start_checkpoint = f"./outputs/NYC/{scene}/chkpnt30000.pth"

    # Base command with environment variables and Python script. The isolated
    # VLM/DLoRAL environments are selected by env vars, so the in-process
    # defaults in arguments/__init__.py pick them up.
    base_cmd = (
        f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
        f"VLM_PYTHON={shlex.quote(VLM_PYTHON)} VLM_MODEL_PATH={shlex.quote(VLM_MODEL_PATH)} "
        f"DLORAL_PYTHON={shlex.quote(DLORAL_PYTHON)} DLORAL_WEIGHT_ROOT={shlex.quote(DLORAL_WEIGHT_ROOT)} "
        f"{shlex.quote(SKYFALL_PYTHON)} train.py"
    )

    # Define base arguments as a list for easy commenting and modification
    base_args = [
        f"-s {dataset_dir}/{scene}",
        f"-m {output_dir}/{model_path}",
        "--eval",
        f"--port {6209+int(gpu)}",
        f"--start_checkpoint {start_checkpoint}",
        "--datasets_type nyc_v1",
    ]

    # Training command
    train_cmd = base_cmd + " " + " ".join(base_args) + " " + command_params
    
    if not dry_run and train:
        success = run_command(train_cmd, log_file, append=True)
        if not success:
            with open(log_file, 'a') as f:
                f.write("\n=== WARNING: Training command failed! ===\n")
            raise RuntimeError(f"GaussianZoom Stage2 failed for {scene}; see {log_file}")

    
    # Log job completion
    with open(log_file, 'a') as f:
        f.write(f"\n=== Job Completed: {model_path} ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return True
        
def worker(gpu, model_path, command_params):
    """Worker function to run a job on a specific GPU."""
    print(f"Starting job on GPU {gpu} with model_path {model_path}")
    print(f"Parameters: {command_params}")
    
    run_job(gpu, model_path, command_params)
    
    print(f"Finished job on GPU {gpu} with model_path {model_path}\n")
    
def dispatch_jobs(jobs, executor):
    """Dispatch jobs to available GPUs."""
    future_to_job = {}
    reserved_gpus = set([])  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            model_path, command_params = job
            
            future = executor.submit(worker, gpu, model_path, command_params)
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job_info = future_to_job.pop(future)
            gpu = job_info[0]
            job = job_info[1]
            reserved_gpus.discard(gpu)  # Release this GPU
            future.result()
            print(f"Job with model_path {job[0]} has finished, releasing GPU {gpu}")
        
        # Small delay to prevent CPU spinning
        time.sleep(5)
        
    print("All jobs have been processed.")

def main():
    # Define jobs as (model_path, command_params) tuples
    jobs = [
        (
            "NYC_004",
            " ".join([
                "--kernel_size 0.1",
                "--resolution 1",
                "--sh_degree 1",
                "--appearance_enabled",
                "--lambda_depth 0.0",
                "--lambda_opacity 10",
                "--opacity_reset_interval 10000000",
                "--iterative_datasets_update",
                "--idu_opacity_reset_interval 5000",
                "--idu_num_samples_per_view 2",
                "--densify_grad_threshold 0.0002",
                "--idu_num_cams 6",
                "--idu_grid_size 4",
                "--idu_grid_width 512",
                "--idu_grid_height 512",
                "--idu_episode_iterations 10000",
                "--idu_opacity_cooling_iterations 500",
                "--lambda_pseudo_depth 0.0",
                "--idu_densify_until_iter 9000",
                "--idu_train_ratio 0.75",
                "--target_std 32",
            ])
        ),
        (
            "NYC_010",
            " ".join([
                "--kernel_size 0.1",
                "--resolution 1",
                "--sh_degree 1",
                "--appearance_enabled",
                "--lambda_depth 0.0",
                "--lambda_opacity 10",
                "--opacity_reset_interval 10000000",
                "--iterative_datasets_update",
                "--idu_opacity_reset_interval 5000",
                "--idu_num_samples_per_view 2",
                "--densify_grad_threshold 0.0002",
                "--idu_num_cams 6",
                "--idu_grid_size 4",
                "--idu_grid_width 512",
                "--idu_grid_height 512",
                "--idu_episode_iterations 10000",
                "--idu_opacity_cooling_iterations 500",
                "--lambda_pseudo_depth 0.0",
                "--idu_densify_until_iter 9000",
                "--idu_train_ratio 0.75",
                "--target_std 32",
            ])
        ),
        (
            "NYC_219",
            " ".join([
                "--kernel_size 0.1",
                "--resolution 1",
                "--sh_degree 1",
                "--appearance_enabled",
                "--lambda_depth 0.0",
                "--lambda_opacity 10",
                "--opacity_reset_interval 10000000",
                "--iterative_datasets_update",
                "--idu_opacity_reset_interval 5000",
                "--idu_num_samples_per_view 2",
                "--densify_grad_threshold 0.0002",
                "--idu_num_cams 6",
                "--idu_grid_size 4",
                "--idu_grid_width 512",
                "--idu_grid_height 512",
                "--idu_episode_iterations 10000",
                "--idu_opacity_cooling_iterations 500",
                "--lambda_pseudo_depth 0.0",
                "--idu_densify_until_iter 9000",
                "--idu_train_ratio 0.75",
                "--target_std 32",
            ])
        ),
        (
            "NYC_336",
            " ".join([
                "--kernel_size 0.1",
                "--resolution 1",
                "--sh_degree 1",
                "--appearance_enabled",
                "--lambda_depth 0.0",
                "--lambda_opacity 10",
                "--opacity_reset_interval 10000000",
                "--iterative_datasets_update",
                "--idu_opacity_reset_interval 5000",
                "--idu_num_samples_per_view 2",
                "--densify_grad_threshold 0.0002",
                "--idu_num_cams 6",
                "--idu_grid_size 4",
                "--idu_grid_width 512",
                "--idu_grid_height 512",
                "--idu_episode_iterations 10000",
                "--idu_opacity_cooling_iterations 500",
                "--lambda_pseudo_depth 0.0",
                "--idu_densify_until_iter 9000",
                "--idu_train_ratio 0.75",
                "--target_std 32",
            ])
        ),
    ]
    
    # Using ThreadPoolExecutor to manage the thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        dispatch_jobs(jobs, executor)

if __name__ == "__main__":
    main()