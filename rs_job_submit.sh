#Single GPU card submission script

#$ -V -cwd

#$ -l h_rt=5:45:00

#$ -l coproc_v100=1

#$ -N "rs_baseline"

#$ -m be

unset DISPLAY
module load python/3.7.4
module load cuda/11.1.1
pip install numpy matplotlib --user
pip install torch --index-url https://download.pytorch.org/whl/cu111 --user
python ./scripts/auto_trainer.py --training_name "rs_baseline" --env_name "RewardShapingEnvironment" --algorithm_name "DDPG" --notes "RS env, DDPG" --batch_size 200 --num_epochs 30001
