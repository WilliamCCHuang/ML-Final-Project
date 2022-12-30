# ML-Final-Project

## Download Datasets

1. ImageNette

    [ImageNette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from Imagenet. It contains 9,469 training images.

    Run the following command to download imagenette160
    
    ```
    $ git clone https://github.com/WilliamCCHuang/ML-Final-Project.git
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

    The accuracy will be printed.

* BYOL

    * Pretrain

        ```
        $ cd ML-Final-Project
        $ sh scripts/byol/train-imagenette-{MODEL}.sh
        ```

        `{MODEL}` should be replaced as one of `resnet18` or `resnet50`.

    * Linear Evaluation

        ```
        $ cd ML-Final-Project
        $ sh scripts/byol/eval-imagenette-{MODEL}.sh
        ```

        `{MODEL}` should be replaced as one of `resnet18` or `resnet50`.