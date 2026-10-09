#!/bin/bash
#SBATCH --job-name=balrog_gptoss_20b
#SBATCH --output=balrog_gptoss_20b_%j.out
#SBATCH --error=balrog_gptoss_20b_%j.err
#SBATCH --partition=L40s_students
#SBATCH --nodelist=netherite
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=32G

set -e

# ---------------------------------------------------------
# Thread limits
# ---------------------------------------------------------
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------
# Hugging Face cache
# ---------------------------------------------------------
export HF_HOME=/local/.cache/huggingface

echo "========================================"
echo "BALROG GPT-OSS-20B"
echo "========================================"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "HF_HOME: $HF_HOME"
echo "========================================"

echo
echo "GPU:"
nvidia-smi

# ---------------------------------------------------------
# Check BALROG installation
# ---------------------------------------------------------
echo
echo "Checking BALROG installation..."

ls -ld /local/s3685918/BALROG
ls -ld /local/s3685918/BALROG/.venv

cd /local/s3685918/BALROG
source .venv/bin/activate

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
MODEL="openai/gpt-oss-20b"
PORT=8084


echo
echo "Model: $MODEL"

# ---------------------------------------------------------
# Start vLLM
# ---------------------------------------------------------
echo
echo "========================================"
echo "Starting vLLM"
echo "========================================"

vllm serve "$MODEL" \
    --port $PORT \
    > vllm_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!

trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "vLLM PID: $VLLM_PID"
echo "Waiting for vLLM..."

# ---------------------------------------------------------
# Wait for vLLM, maximum 5 minutes
# ---------------------------------------------------------
for i in $(seq 1 60); do

    if curl -sf http://localhost:$PORT/v1/models > /dev/null; then
        echo "vLLM is ready!"
        break
    fi

    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo
        echo "ERROR: vLLM process died."
        echo
        echo "========== vLLM LOG =========="
        cat vllm_${SLURM_JOB_ID}.log
        echo "=============================="
        exit 1
    fi

    sleep 5
done

if ! curl -sf http://localhost:$PORT/v1/models > /dev/null; then
    echo
    echo "ERROR: vLLM did not become ready within 5 minutes."
    echo
    echo "========== vLLM LOG =========="
    cat vllm_${SLURM_JOB_ID}.log
    echo "=============================="
    exit 1
fi

echo
echo "========================================"
echo "vLLM model"
echo "========================================"

curl -s http://localhost:$PORT/v1/models

# ---------------------------------------------------------
# Run BALROG
# ---------------------------------------------------------
echo
echo "========================================"
echo "Running BALROG"
echo "========================================"

python eval.py \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=vllm \
    client.model_id="$MODEL" \
    client.base_url=http://localhost:$PORT/v1 \
    eval.resume_from=results/2026-10-08_18-10-17_naive_openai_gpt-oss-20b
echo
echo "========================================"
echo "BALROG finished successfully"
echo "========================================"
