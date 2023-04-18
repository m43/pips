from pips_utils.util import ensure_dir

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --chdir /scratch_net/biwidl217/frrajic/eth-master-thesis/03-code/pips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o ./logs/slurm_logs/%x-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
source /scratch_net/biwidl217/frrajic/miniconda3/etc/profile.d/conda.sh
conda activate pips

# Run
date
printf "The run has been configured and the environment set up. Gonna run the experiment now.\\n\\n"

export WANDB__SERVICE_WAIT=300
{command}

# Wait for all background processes to finish
echo "Main process waiting for background processes to finish..."
FAIL=0
for job in $(jobs -p); do
  echo Waiting for: $job
  wait $job || let "FAIL+=1"
done
echo $FAIL
if [ "$FAIL" == "0" ]; then
  echo "All background processes finished successfully."
else
  echo "$FAIL background processes DID NOT finish successfully."
fi
echo "Done."

echo FINISHED at $(date)
"""

command_list = [
    "python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
    "python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output",
]

if __name__ == '__main__':
    ensure_dir("./sbatch_files")
    for i, command in enumerate(command_list):
        with open(f"./sbatch_files/A1_{i + 1:03d}.sh", "w") as f:
            f.write(SLURM_TEMPLATE.format(command=command))
