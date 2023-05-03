#Single GPU card submission script

#$ -V -cwd

#$ -l h_rt=2:35:00

#$ -l coproc_v100=1

#$ -N "gr_env_ic3"

#$ -m be

unset DISPLAY
module load python/3.7.4
module load cuda/11.1.1
pip install numpy matplotlib --user
pip install torch --index-url https://download.pytorch.org/whl/cu111 --user
python ./scripts/auto_trainer.py --training_name "gr_env_ic3" --env_name "GaussianRewardEnvironment" --algorithm_name "JIT_DDPG" --notes "gr_env_ic3" --batch_size 200 --num_epochs 5001 --alpha 0.00008 --beta 0.0008 --gamma 0.95 --sigma 0.2 --tau 0.01 --initial_position True