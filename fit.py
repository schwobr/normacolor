"""
Script to fit autoencoder.

**************************
"""
# coding: utf8
import argparse
from pathlib import Path
from data import get_items, get_generators, getNextFilePath
from model import make_autoencoder_model
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str,
                    help="path to png dataset directory.")
parser.add_argument("--outdir", type=str,
                    help="path to output directory.")
parser.add_argument("--device", default="1",
                    help="ID of the device to use for computation.")
parser.add_argument("--epochs", type=int, default=20,
                    help="number of epochs for training the model.")
parser.add_argument("--batch-size", type=int, default=128,
                    help="number of samples in one batch for fitting.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate, for the SGD optimizer.")
parser.add_argument("--ratio", type=float, default=1e-3,
                    help="percentage of patches to keep per image")
parser.add_argument("--patch-size", type=int, default=15,
                    help="size of the patches to extract")
parser.add_argument("--width", type=int, default=16,
                    help="number of filters in first layer.")
parser.add_argument("--depth", type=int, default=3,
                    help="number of encoding layers.")

args = parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    itemlist = get_items(Path(args.dataset), extensions=['.png'])
    train_gen, val_gen = get_generators(
        itemlist, args.batch_size, args.ratio, args.patch_size)
    autoencoder = make_autoencoder_model(
        width=args.width, depth=args.depth, patch_size=args.patch_size)
    history = autoencoder.fit_generator(train_gen,
                                        epochs=args.epochs,
                                        validation_data=val_gen,
                                        steps_per_epoch=len(train_gen),
                                        validation_steps=len(val_gen))
    exp_id = getNextFilePath(Path(args.outdir), "history")
    autoencoder.save_weights(os.path.join(
        args.outdir, "weights_{}.h5".format(exp_id)))
    file_path = os.path.join(getNextFilePath(
        args.outdir, "history_{}.p".format(exp_id)))
    with open(file_path, "wb") as f:
        pickle.dump(history, f)
