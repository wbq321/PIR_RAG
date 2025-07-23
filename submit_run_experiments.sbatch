#!/bin/bash
#SBATCH --job-name=pir_rag_exp   # 作业名称
#SBATCH --output=pir_rag_exp_%j.out # 输出文件名, %j 会被作业ID替换
#SBATCH --error=pir_rag_exp_%j.err  # 错误文件名
#SBATCH --time=12:00:00          # 预计运行时间 (4小时)
#SBATCH --nodes=1                # 使用1个节点
#SBATCH --ntasks-per-node=1      # 每个节点上运行1个任务
#SBATCH --cpus-per-task=16       # 为任务申请16个CPU核心
#SBATCH --mem=64G                # 申请64GB内存

echo "Job started on $(hostname) at $(date)"
cd $SLURM_SUBMIT_DIR
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate pir_rag_env

# 运行你的实验脚本
# 使用 -u 选项来确保Python的输出是无缓冲的，这样你可以实时在.out文件中看到日志
python -u run_experiments.py \
    --embeddings_path msmarco_data/embeddings_10000.npy \
    --corpus_path msmarco_data/corpus_10000.csv

echo "Job finished at $(date)"
