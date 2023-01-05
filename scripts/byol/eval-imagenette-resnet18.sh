python eval.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --checkpoint-path exps/byol/resnet18/byol_learner.pt \
    --evaluation-scheme linear \
    --num-layers 18 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.001 \
    --output-dir exps/byol/resnet18 \
    --gpu-idx 0