CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py \
  --dataset msmt17 \
  --batch-size 256 \
  --iters 200 \
  --eps 0.7 \
  --self-norm \
  --use-hard \
  --num-instances 8 \
  --eval-step 50 \
  --logs-dir logs/msmt17_sg
