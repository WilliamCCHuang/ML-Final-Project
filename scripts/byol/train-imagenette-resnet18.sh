python train.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --training-scheme byol \
    --num-layers 18 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 1e-6 \
    --output-dir exps/byol/resnet18 \
    --gpu-idx 0