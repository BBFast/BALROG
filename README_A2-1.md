For exercise 1, the following models are tested with vLLM:

- meta-llama/Llama-3.2-1B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- openai/gpt-oss-20b

Additionally, one cloud model is used via an API key:
- gpt-4o-mini-2024-07-18

For this exercise we only make use of the provided `balrog/agents/naive.py` agent.

The following command is used to run the Llama model:

```bash
python eval.py \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=vllm \
    client.model_id=meta-llama/Llama-3.2-1B-Instruct \
    client.base_url=http://localhost:8080/v1
```

The following scripts can be run using `bash` (or `sbatch` when using a SLURM machine): qwen2.5_3B.sh, qwen2.5_7B.sh, qwen2.5_14B.sh, gpt_oss_20b.sh.

To run the gpt-4o-mini-2024-07-18 model the following command is used:

```bash
python eval.py \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=openai \
    client.model_id=gpt-4o-mini-2024-07-18
```