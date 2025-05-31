import subprocess
import time

n_chunks = 93  # Total number of redshift chunks
script_path = "learn_f.py"

for z_idx in range(0, n_chunks, 2):
    processes = []

    # Launch even chunk on GPU 0
    cmd0 = ["python", script_path, str(z_idx), "0"]
    print(f"Launching chunk {z_idx} on GPU 0")
    p0 = subprocess.Popen(cmd0)
    processes.append(p0)

    # Launch odd chunk on GPU 1, if within range
    if z_idx + 1 < n_chunks:
        cmd1 = ["python", script_path, str(z_idx + 1), "1"]
        print(f"Launching chunk {z_idx + 1} on GPU 1")
        p1 = subprocess.Popen(cmd1)
        processes.append(p1)

    # Wait for both processes to finish before moving on
    for p in processes:
        p.wait()

    print(f"Finished pair: {z_idx} and {z_idx + 1 if z_idx + 1 < n_chunks else 'None'}\n")