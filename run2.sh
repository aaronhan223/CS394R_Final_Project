CUDA_VISIBLE_DEVICES=2 python3.6 main.py --env-name 'HalfCheetah-v2' --algo ppo --use-gae --log-interval 10 --num-steps 2048 --seed $1 --num-processes 1 --lr 0.0003 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --dimh 32 --use-linear-lr-decay --use-proper-time-limits

