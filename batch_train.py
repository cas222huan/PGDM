# batch_train.py
import subprocess
import sys
import time

models = ["resshift_v1_10x", "resshift_v1_20x"] # change as you need [must match the names in configs/var.yaml]
types = ["resshift"] * len(models) # change as you need ["resshift" for PGDM and "mocolsk" for MoCoLSK-Net]
# train each model iteratively
for i, model in enumerate(models):
    print(f"\n{'='*50}")
    print(f"Model [{i+1}/{len(models)}]: {model}")
    print(f"{'='*50}")
    
    try:
        # run the training script with the specified model
        result = subprocess.run(
            [sys.executable, "train.py", "--var", model, "--type", types[i]],
            check=True
        )

        print(f"Model {model} has been trained.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.returncode} | {e.stderr}")

    if i < len(models) - 1:
        print(f"Wait 5 seconds for the next model...")
        time.sleep(5)

print(f"All models have been trained.")