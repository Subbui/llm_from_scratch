import urllib.request

from gpt_download import download_and_load_gpt2

# url = (
# "https://raw.githubusercontent.com/rasbt/"
# "LLMs-from-scratch/main/ch05/"
# "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)
# 
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print(settings)
print(params.keys())