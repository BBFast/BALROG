For exercise 2, the following models are tested with vLLM:

- meta-llama/Llama-3.2-1B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct

These models are tested on the following agents: `balrog/agents/custom.py`, `balrog/agents/custom_cot_sbs.py`, `balrog/agents/custom_cot_prevact.py`. Here, `custom_cot_sbs.py` is the agent implemented with "Think step by step". `custom_cot_prevact.py` is the agent which explicitly gives the previous action and provide steps to think about in the prompt.

The following command is used to run the Llama model, for `$AGENT = [custom, custom_cot_sbs, custom_cot_prevact]`:

```bash
python eval.py \
    agent.type=$AGENT \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=vllm \
    client.model_id=meta-llama/Llama-3.2-1B-Instruct \
    client.base_url=http://localhost:8080/v1
```

The following scripts can be run using `bash` (or `sbatch` when using a SLURM machine) which will run all agents sequentially: qwen2.5_3B_A2.sh, qwen2.5_7B_A2.sh