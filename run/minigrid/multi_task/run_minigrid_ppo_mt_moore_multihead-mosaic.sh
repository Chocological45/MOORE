#!/bin/bash

cd ../../../

ENV_NAME=$1
N_EXPERTS=$2

python run_minigrid_ppo_mt.py  --n_exp 5 \
                            --env_name ${ENV_NAME} --exp_name ppo_mt_moore_multihead_${N_EXPERTS}e \
                            --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
                            --critic_network MiniGridPPOMixtureMHNetwork --critic_n_features 128 --orthogonal --n_experts ${N_EXPERTS} \
                            --actor_network MiniGridPPOMixtureMHNetwork --actor_n_features 128 \
                            --batch_size 256 --gamma 0.99 --seed 9157 9802 9822 2211 1911 \
                            --wandb --wandb_entity [saptarshinath12-loughborough-university]