#!/bin/bash
#SBATCH -p kyutai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c 224
#SBATCH --mem=256G
#SBATCH -t 2:30:00
#SBATCH --exclusive
#SBATCH -J e2e_3stacks
#SBATCH --exclude=par2dc5-ai-prd-cl02s03dgx01
#SBATCH -o /home/gabriel/ddp_work/logs/e2e_three_stacks_%j.log
# nanoGPT DDP, 16 ranks on 2 nodes, 30 steps, three stacks. Per batch size: one discarded
# warm-up run, then five interleaved rounds (ABC CBA ABC CBA ABC). Every run carries the
# evidence the summariser checks: NCCL's INIT/NET debug lines (NCCL stacks), the loader's
# "collectives via ..." trace line (mojo stacks), and rank_bind's NUMA binding lines.
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1); export MASTER_ADDR
scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '; echo
W=${MOJO_TREE:-/home/gabriel/ddp_work/autocast_wt}; cd $W; echo "mojo tree: $(git log --oneline -1)"
NANOGPT=/home/gabriel/ddp_work/nanoGPT
STOCK_TR=/home/gabriel/ddp_work/torch_cu128/bin/torchrun
NCCLDBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
for BS in ${BATCHES:-48 32 12}; do
  COMMON="--nanogpt-path $NANOGPT --data-dir $NANOGPT/data/shakespeare --log-interval 1 --seed 1337 --max-iters 30 --eval-interval 0 --batch-size $BS"
  i=0
  for cfg in warmup stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl; do
    i=$((i+1)); LOG=/home/gabriel/ddp_work/logs/e2e_three_stacks_${SLURM_JOB_ID}_bs${BS}_$(printf %02d $i)_$cfg.log
    case $cfg in
      warmup|stock) TR="$STOCK_TR"; DEV=cuda; ENV="PYTHONPATH= $NCCLDBG";;
      mojo_nccl)    TR="uv run --no-sync torchrun"; DEV=mojo; ENV="$NCCLDBG";;
      mojo_mojoccl) TR="uv run --no-sync torchrun"; DEV=mojo; ENV="TORCH_MOJO_BACKEND_CCL=mojo";;
    esac
    echo "=== bs=$BS run $i cfg=$cfg ==="
    srun bash -c "cd $W && env PYTHONPATH=$W $ENV flock /tmp/gpu_lock_0.lock $TR --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29851 --rdzv-id=\$SLURM_JOB_ID-$BS-$i /home/gabriel/ddp_work/autocast_wt/tests/multinode/rank_bind.py --device $DEV $COMMON" > $LOG 2>&1
    echo "=== bs=$BS run $i exit: $? ==="
  done
done
echo "=== HONEST DONE ==="
