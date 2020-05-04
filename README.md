# Crowd Density Estimation

A simple crowd density baseline models using Pytorch.

BLOG: https://katnoria.com/crowd-density/

## Packages Required
pytorch, numpy, matplotlib, pandas, tensorboardX and tensorboard

# Checkpoints
You can download the model checkpoints using `download_checkpoints.sh` script.

# Training

Option 1: Use the train-models notebook under the notebooks folder to train the model.

Option 2: Use the trainer.py script directly to train the model

Use `python trainer.py --help`

Example: `python trainer.py --model vggdecoder --cropsize 224 --epochs 400 --lr 1e-5`

Use tensorboard to see the training and test plots. Make sure you start the tensorboard from the correct folder.
If you using the notebook to train the model, you'll need to start tensorboard from notebooks folder.

`tensorboard --logdir runs --host 0.0.0.0`

# Testing

Make sure you have downloaded the model checkpoints or have locally trained the model.
If you have trained your model locally, you should use the `--checkpoint` flag to specify its path.

## VGG Baseline

Evaluate single image 
`
python tester.py --checkpoint "src/ckpt/vgg16baseline_448_400ep_private-1e-05-0.052820200430.pt" --single ../eval_imgs/has_136_heads.jpg
`

Evaluate the single image and save the output image. The output image overlays the density map over the input image.

`python tester.py --checkpoint "src/ckpt/vgg16baseline_448_400ep_private-1e-05-0.052820200430.pt" --imagepath "../eval_imgs/*.jpg" --save True`

Evaluate the directory full of images

`python tester.py --checkpoint "src/ckpt/vgg16baseline_448_400ep_private-1e-05-0.052820200430.pt" --imagepath "../eval_imgs/*.jpg" --save True`

## VGG Decoder (448)

Similarly you can use vggdecoder model to evaluate the images.

`
python tester.py --model vggdecoder --checkpoint "src/ckpt/vgg16decoder_448_400ep_private-1e-05-0.050320200501.pt" --single "/mnt/bigdrive/images/crowd-density.jpg" --save True
`