CUDA_VISIBLE_DEVICES=2 nohup python -u main.py \
--model resnet20_1w1a \
--results_dir ./ \
--save result \
--epoch 405 \
--lr 0.1 \
-b 1024 \
> resnet20_L1.log 2>&1 &
