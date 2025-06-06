# driver_train.py

import subprocess

num_models = 10
for i in range(0, num_models, 2):
    procs = []
    for j, gpu_id in zip(range(i, i + 2), ["0", "1"]):
        if j >= num_models:
            break
        print(f"Launching model {j} on GPU {gpu_id}")
        p = subprocess.Popen(["python", "train_Hz.py", str(j), gpu_id])
        procs.append(p)
    
    # Wait for both to finish before continuing
    for p in procs:
        p.wait()