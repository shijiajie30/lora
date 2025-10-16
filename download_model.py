from modelscope.hub.snapshot_download import snapshot_download

model_name = 'LLM-Research/Llama-3.2-3B-Instruct'
model_path = snapshot_download(model_name, cache_dir='./models', revision='master')