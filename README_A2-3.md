For exercise 3, the following models are tested with vLLM:

- meta-llama/Llama-3.2-1B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct

and one LLM via the API:
- gpt-4o-mini-2024-07-18

These models are tested on the following agents: `balrog/agents/custom_reflexion.py` and `balrog/agents/custom_react_babyAI.py`. `custom_reflexion.py` implements the Reflexion-like agent. `custom_react_babyAI.py` implements the ReAct agent on babyAI. The Reflexion-like agent is evaluated on BabyAI, BabaIsAI, TextWorld and Crafter. The ReAct agent only on BabyAI. 

The following command is used to run the Llama model, for `$AGENT = [custom_reflexion, custom_react_babyAI]` and `$ENVS = babyai-babaisai-textworld-crafter` for custom_reflexion and `babyai` for custom_react_babyAI:

```bash
python eval.py \
    agent.type=$AGENT \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=vllm \
    client.model_id=meta-llama/Llama-3.2-1B-Instruct \
    client.base_url=http://localhost:8080/v1 \
    envs.names=$ENVS
```
Additionally, gpt-4o-mini-2024-07-18 can be run using the command:

```bash
python eval.py \
    agent.type=custom_react_babyAI \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=openai \
    client.model_id=gpt-4o-mini-2024-07-18 \
    envs.names=babyai
```

The following scripts can be run using `bash` (or `sbatch` when using a SLURM machine) which will run all agents sequentially: qwen2.5_3B_A3.sh, qwen2.5_7B_A3.sh which run `custom_reflexion.py` and qwen2.5_3B_A3_react.sh, qwen2.5_7B_A3_react.sh which run `custom_react_babyAI.py`