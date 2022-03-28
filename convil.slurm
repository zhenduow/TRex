#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1
#SBATCH --mail-user=zhenduow94@gmail.com
#SBATCH --mail-type=FAIL,END

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant


export CUDA_VISIBLE_DEVICES=2
srun python3.6 run.py \
--dataset_name UDC-ActionParsnip \
--cv 0 \
--reranker_name poly \
--user_tolerance 1 \
--user_patience 10 \
--batch_size 100 \
--lr 1e-4 \
--lrdc 0.98 \
--cq_reward 0.11 \
--cascade_p 0.5 \
--disc_train_ratio 5 \
--weight_decay 0.01 \
--il_topn 3 \
--max_data_size 10000 \
--max_iter 1000 \
--gan_name TRexPPO \
--risk_run_epoch 20 \
--base_run_epoch 20 \
--dropout_ratio 0.2 \
--trex_n_rewards 5 \
--trex_traj_sample_length 1
#--load_checkpoint True \
#--checkpoint_il il_31.666705399751663
