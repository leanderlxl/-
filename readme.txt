python train.py \
  --train-img-dir /root/autodl-tmp/train2014\
  --train-ann-file /root/autodl-tmp/annotations/instances_train2014.json \
  --val-img-dir /root/autodl-tmp/val2014\
  --val-ann-file /root/autodl-tmp/annotations/instances_val2014.json \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-5 \
  --device cuda
