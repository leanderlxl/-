python train.py \
  --train-img-dir /root/autodl-tmp/train2014\
  --train-ann-file /root/autodl-tmp/annotations/instances_train2014.json \
  --val-img-dir /root/autodl-tmp/val2014\
  --val-ann-file /root/autodl-tmp/annotations/instances_val2014.json \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-3 \
  --device cuda
明天早上起来再继续训练5轮
早上起来先拉取代码，然后尽量不要停止训练，因为下午不好抢
显卡

然后训他个十五轮