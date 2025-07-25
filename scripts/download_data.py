# download_and_prepare.py
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

# 确保缓存目录设置正确
cache_dir = "/scratch/user/u.bw269205/huggingface_cache"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

print(f"Hugging Face cache directory set to: {cache_dir}")

def download_and_extract(sample_size=1000000, out_dir="msmarco_data_prepared"):
    """
    在有网络的登录节点运行。
    1. 下载并缓存所有资源。
    2. 加载数据集，抽样，并提取文本保存到简单的CSV文件中。
    """
    print("--- Phase 1: Downloading and Caching Assets ---")
    
    # 1. 下载模型
    model_name = 'BAAI/bge-base-en-v1.5'
    try:
        print(f"Downloading and caching model: {model_name}...")
        SentenceTransformer(model_name, cache_folder=cache_dir)
        print("Model cached successfully.")
    except Exception as e:
        print(f"Could not download model. It might already be fully cached. Error: {e}")
        
    # 2. 下载数据集
    dataset_name = "ms_marco"
    config_name = "v1.1"
    try:
        print(f"Downloading and caching dataset: {dataset_name} ({config_name})...")
        load_dataset(dataset_name, config_name, split='train', cache_dir=cache_dir)
        print("Dataset cached successfully.")
    except Exception as e:
        print(f"Could not download dataset. It might already be fully cached. Error: {e}")
        
    print("\n--- Phase 2: Extracting and Saving Corpus ---")
    
    # 3. 加载、抽样并保存为简单格式
    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, f"corpus_{sample_size}.csv")

    if os.path.exists(corpus_path):
        print(f"Corpus file {corpus_path} already exists. Skipping extraction.")
    else:
        print(f"Loading dataset from cache to extract {sample_size} passages...")
        dataset = load_dataset(dataset_name, config_name, split='train', cache_dir=cache_dir)
        shuffled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        print("Extracting passage texts...")
        corpus_texts = [item['passages']['passage_text'] for item in shuffled_dataset]
        
        print(f"Saving extracted texts to {corpus_path}")
        corpus_df = pd.DataFrame({'text': corpus_texts})
        corpus_df.to_csv(corpus_path, index=False)

    print("\nPreparation on login node complete.")
    print("You can now submit the embedding generation job.")

if __name__ == "__main__":
    download_and_extract(sample_size=10000)