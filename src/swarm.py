# swarm.py
import subprocess
from pathlib import Path
from src.config import BASE_PATH, SEED


def get_swarm_time(minutes):
    if minutes <= 240:
        partition = "quick"
    else:
        partition = "norm"
    hours, mins = divmod(minutes, 60)
    if hours == 0:
        mins = max(10, mins)
    # Format as "HH:MM:SS"
    time_string = f"{hours:02d}:{mins:02d}:00"
    return time_string, partition


####################################################
################## SWARM SPECIFIC ##################
####################################################
def run_tune_swarm(*_, model_list, swarm_log_dir, cmd_dir, n_jobs, n_threads, n_trials):
    for model_abrv in model_list:
        num_threads = 2 * n_jobs * n_threads
        if model_abrv == "nn":
            # needs a bit more resources
            num_threads += 4
            tot_mins = n_trials * 3
            gb = 5
        elif model_abrv == "svc":
            tot_mins = n_trials * 3
            gb = 5
        elif model_abrv in ["xgb", "lgbm", "lr"]:
            tot_mins = n_trials / 2
            gb = 5
        else:
            raise ValueError(f"Unrecognized model_abrv: {model_abrv}")

        swarm_time, partition = get_swarm_time(int(tot_mins))

        log_dir = swarm_log_dir / model_abrv
        log_dir.mkdir(parents=True, exist_ok=True)

        swarm_path = cmd_dir / f"{model_abrv}.swarm"
        swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {num_threads} --logdir={log_dir}  --job-name={model_abrv.upper()} --partition={partition} {swarm_path}"
        print(swarm_cmd)
        subprocess.run(swarm_cmd, shell=True)


def run_swarms_eval(swarm_log_dir, swarm_path, n_jobs, n_bootstraps):
    num_threads = max(2 * n_jobs, 8)
    num_mins = int(n_bootstraps / 25)

    swarm_time, partition = get_swarm_time(num_mins)
    gb = 5
    swarm_log_dir.mkdir(parents=True, exist_ok=True)
    swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {num_threads} --logdir={swarm_log_dir}  --job-name=EVAL --partition={partition} {swarm_path}"
    print(swarm_cmd)
    subprocess.run(swarm_cmd, shell=True)


def run_swarms_shap(swarm_log_dir, cmd_dir, model_list):
    for model_name in model_list:
        swarm_path = cmd_dir / f"{model_name}.swarm"
        log_dir = swarm_log_dir / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        if model_name in ["svc", "stack", "nn"]:
            swarm_time = "12:00:00"
            partition = "norm"
            batch_size = 1
        else:
            swarm_time = "0:30:00"
            partition = "quick"
            batch_size = 6

        swarm_cmd = f"swarm --time={swarm_time} -g {12} -t 6 --logdir={log_dir}  --job-name=SHAP-{model_name.upper()} --partition={partition} -b {batch_size} {swarm_path}"
        print(swarm_cmd)
        subprocess.run(swarm_cmd, shell=True)
