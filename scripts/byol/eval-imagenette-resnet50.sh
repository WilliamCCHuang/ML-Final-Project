python eval.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --checkpoint-path exps/byol/resnet50/byol_learner.pt \
    --evaluation-scheme linear \
    --num-layers 50 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.001 \
    --output-dir exps/byol/resnet50 \
    --gpu-idx 0