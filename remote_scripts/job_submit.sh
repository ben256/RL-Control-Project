#Single GPU card submission script

#$ -V -cwd

#$ -l h_rt=3:50:00

#$ -l coproc_v100=1

#$ -N "RO_RM_14"

#$ -m be

unset DISPLAY
module load python/3.7.4
module load cuda/11.1.1
pip install numpy matplotlib --user
pip install torch --index-url https://download.pytorch.org/whl/cu111 --user
python ./scripts/auto_trainer.py --training_name "RO_RM_14" --env_name "RewardMachineEnvironment" --algorithm_name "DDPG" --notes "Changed RM2, updated dynamic equations" --rm "rm4" --batch_size 200 --num_epochs 20000
