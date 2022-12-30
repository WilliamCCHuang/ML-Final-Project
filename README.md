# ML-Final-Project

## Download Datasets

1. ImageNette

    [ImageNette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from Imagenet. It contains 9,469 training images.

    Run the following commands to download imagenette160
    
    ```
    $ cd ML-Final-Project/data
    $ wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    $ tar zxvf imagenette2-160.tgz
    ```

    Once done, there should be a directory called `imagenette2-160` in the directory `data`.

## Run Experiments

* Supervised leanring

    ```
    $ cd ML-Final-Project
    $ sh scripts/supervised/train-imagenette-{MODEL}.sh
    ```

    `{MODEL}` should be replaced as one of `resnet18` or `resnet50`.

    The model weight will be saved in the directory `exps/supervised/{MODEL}/model.pt`.
    
    The top-1 accuracy will be printed.

* BYOL

    * Pretrain

        ```
        $ cd ML-Final-Project
        $ sh scripts/byol/train-imagenette-{MODEL}.sh
        ```

        `{MODEL}` should be replaced as one of `resnet18` or `resnet50`.

        The model weight will be saved in the directory `exps/byol/{MODEL}/byol_leaner.pt`.

    * Linear Evaluation

        ```
        $ cd ML-Final-Project
        $ sh scripts/byol/eval-imagenette-{MODEL}.sh
        ```

        `{MODEL}` should be replaced as one of `resnet18` or `resnet50`.

        The top-1 accuracy will be printed.

## Explanations

1. Because BYOL needs two augmented views of an image, I implemented two augmentations in the class `ImageNetteDataset`. See [here]((https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/datasets.py#L55)) for detail.

2. The BYOL algorithm is implemented in the class [`BYOL`](https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/byol.py#L32). One can [input](https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/byol.py#L100) two augmented views of a batch from a data loader, and then return BYOL loss by calling the method [`self._compute_loss()`](https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/byol.py#L77).

    In the method `self._compute_loss()`, the online network that consists of an encoder, a projector, and a predictor, is inputted with `x1` , and outputs the prediction $q_\theta(z_\theta)$, while `x2` is projected into the target $z'_\xi$ by the target network that only has an encoder and a projector. The class [BYOLLoss](https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/losses.py#L5) uses the prediction $q_\theta(z_\theta)$ and the target $z'_\xi$ to compute BYOL loss.

    The most important thing is that only the online network is updated by back propagation. The target network is updated with a slow-moving average of the online network. It is done by the method [`self.update_target_network()`](https://github.com/WilliamCCHuang/ML-Final-Project/blob/main/byol.py#L110). This method is called after the online network is updated.

## Requirements

See the file `requirements.txt`.
