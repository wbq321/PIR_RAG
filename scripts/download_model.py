# download_model.py (确保它是这样的)
from sentence_transformers import SentenceTransformer
import os

# 定义模型名和你希望保存的本地路径 (一个简单的、在你项目目录下的文件夹名)
model_name = 'BAAI/bge-base-en-v1.5'
local_model_path = './local_bge_model' 

# 检查目录是否已存在，如果存在则跳过，避免重复下载
if os.path.exists(local_model_path):
    print(f"Local model already exists at: '{local_model_path}'. Skipping download.")
else:
    print(f"Downloading model '{model_name}' from Hugging Face Hub...")
    # 这一步会从网上下载模型到默认的缓存目录
    model = SentenceTransformer(model_name)
    
    print(f"Saving model to a clean local directory: '{local_model_path}'...")
    # 这一步会将模型从缓存中复制出来，整理成一个可直接加载的、干净的文件夹结构
    model.save(local_model_path)

    print("\nModel saved successfully in a local directory.")
    print(f"You should now use '{local_model_path}' as the path in your experiment script.")