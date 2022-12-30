python eval.py \
    --img-dir data/imagenette2-160 \
    --img-size 128 \
    --checkpoint-path exps/byol/resnet18/byol_learner.pt \
    --evaluation-scheme linear \
    --num-layers 18 \
    --epochs 100 \
    --batch-size 128 \
    --output-dir exps/byol/linear \
    --gpu-idx 1