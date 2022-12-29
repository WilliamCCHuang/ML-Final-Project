python train.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --training-scheme byol \
    --num-layers 18 \
    --epochs 100 \
    --batch-size 64 \
    --lr-base 0.02 \
    --output-dir exps/byol/resnet18 \
    --gpu-idx 1