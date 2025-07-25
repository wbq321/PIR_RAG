# generate_embeddings.py (最终版)
import os
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------- 釜底抽薪的解决方案 -----------------
# 我们不再依赖环境变量，而是直接找到模型的本地路径
def find_model_path(model_name: str, cache_dir: str) -> str:
    """
    在Hugging Face缓存目录中找到模型的本地文件夹路径。
    """
    # 将 'BAAI/bge-base-en-v1.5' 转换为 'models--BAAI--bge-base-en-v1.5'
    model_folder_name = "models--" + model_name.replace("/", "--")
    model_path_pattern = os.path.join(cache_dir, model_folder_name, "snapshots", "*")
    
    # 查找快照文件夹
    snapshots = glob.glob(model_path_pattern)
    if not snapshots:
        raise FileNotFoundError(
            f"Model snapshot not found for '{model_name}' in cache '{cache_dir}'. "
            "Please run the download script on a node with internet access first."
        )
    
    # 通常只有一个快照，我们取最新的一个
    model_local_path = max(snapshots, key=os.path.getmtime)
    print(f"Found model's local cache path: {model_local_path}")
    return model_local_path
# --------------------------------------------------------


def main(sample_size=10000, data_dir="msmarco_data_prepared"):
    print(f"--- Starting Embedding Generation for {sample_size} docs ---")
    
    cache_dir = "/scratch/user/u.bw269205/huggingface_cache"
    
    # 1. 从我们自己创建的CSV加载语料库
    corpus_path = os.path.join(data_dir, f"corpus_{sample_size}.csv")
    print(f"Loading corpus from: {corpus_path}")
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus file not found. Please run the preparation script on the login node first.")
        return
        
    corpus_df = pd.read_csv(corpus_path)
    corpus_texts = corpus_df['text'].dropna().tolist()
    
    # 2. 从本地缓存加载模型 (使用新的方法)
    print("Initializing sentence transformer model from a specific local path...")
    model_name = 'BAAI/bge-base-en-v1.5'
    
    try:
        # 直接找到模型的本地文件夹路径
        local_model_path = find_model_path(model_name, cache_dir)
        # 将这个本地路径传递给SentenceTransformer
        model = SentenceTransformer(local_model_path, device='cpu')
    except FileNotFoundError as e:
        print(e)
        return

    # 3. 生成嵌入
    print("Generating embeddings...")
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f"Using up to {cpu_count} CPU cores.")
    embeddings = model.encode(
        corpus_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        device='cpu',
        num_workers=cpu_count - 1 if cpu_count > 1 else 0
    )
    
    # 4. 保存嵌入
    embeddings_path = os.path.join(data_dir, f"embeddings_{sample_size}.npy")
    print(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))

    print("\nEmbedding generation complete.")

if __name__ == "__main__":
    main(sample_size=10000)