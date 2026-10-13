


benchmarks: BabyAI - babaisai - crafter - Textworld (?) 


models: Qwen 3B - Qwen 7B - Llama 1B proberen te verbeteren naar niveau gpt oss. Als runtime het toelaat ook 14B proberen te verhogen ivm slechte score

JobID 21900 (oude versie van custom_cot.py degene zonder previous action included (commit small change to try to prevent repetitive actions)

21912 + 21913 zijn met previous action gecombineerd (commit added previous action to prompt)

For Llama 1B: python eval.py   agent.type=custom   agent.max_image_history=0   agent.max_text_history=16   eval.num_workers=8   client.client_name=vllm   client.model_id=meta-llama/Llama-3.2-1B-Instruct   client.base_url=http://0.0.0.0:8081/v1 envs.names=babyai-babaisai-textworld-crafter