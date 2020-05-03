nohup python -u main.py \
--gpus 0,1,2 \
--model resnet18_1w1a \
--results_dir ./ \
--save result \
--data_path /media/disk2/zyc/ImageNet2012 \
--dataset imagenet \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
> output.log 2>&1 &
