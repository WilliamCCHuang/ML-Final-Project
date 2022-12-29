python train.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --training-scheme supervised \
    --num-layers 18 \
    --epochs 1000 \
    --batch-size 64 \
    --lr-base 0.004 \
    --output-dir exps/supervised/resnet18 \
    --gpu-idx 0