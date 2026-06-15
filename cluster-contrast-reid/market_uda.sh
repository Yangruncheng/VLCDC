# Optional UDA-style run after preparing a source-domain checkpoint.
CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py \
  --dataset market1501 \
  --batch-size 256 \
  --iters 200 \
  --eps 0.6 \
  --self-norm \
  --use-hard \
  --num-instances 8 \
  --logs-dir logs/msmt17_to_market1501_sg
