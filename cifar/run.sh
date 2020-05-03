nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir ./ \
--save result \
--data_path /home/xuzihan/data \
--dataset cifar10 \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
> output.log 2>&1 &
