#!/bin/bash
#SBATCH --job-name=prepare_msmarco   # 作业名称
#SBATCH --output=prepare_msmarco_%j.out # 输出日志 (%j = 作业ID)
#SBATCH --error=prepare_msmarco_%j.err  # 错误日志
#SBATCH --time=4:00:00          # 预计运行时间 (12小时，这个任务很长)
#SBATCH --nodes=1                # 使用1个节点
#SBATCH --ntasks-per-node=1      # 每个节点上运行1个任务
#SBATCH --cpus-per-task=16       # 申请16个CPU核心
#SBATCH --mem=128G               # 申请128GB内存 (非常重要!)

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "======================================================"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate pir_rag_env
# 运行你的数据准备脚本
python -u generate_embeddings.py # -u 确保输出不被缓冲

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"
