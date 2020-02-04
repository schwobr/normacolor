# coding: utf8
from model import make_autoencoder_model
from data import get_items, getNextFilePath
from cluster import get_centroids, get_histograms
import argparse
from pathlib import Path
from keras.models import Model
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str,
                    help="path to png dataset directory.")
parser.add_argument("--outdir", type=str,
                    help="path to output directory.")
parser.add_argument("--device", default="1",
                    help="ID of the device to use for computation.")
parser.add_argument("--batch-size", type=int, default=128,
                    help="number of samples in one batch for fitting.")
parser.add_argument("--patch-size", type=int, default=16,
                    help="size of the patches to extract.")
parser.add_argument("--img-size", type=int, default=256,
                    help="sizes of the original images.")
parser.add_argument("--width", type=int, default=16,
                    help="number of filters in first layer.")
parser.add_argument("--depth", type=int, default=3,
                    help="number of encoding layers.")
parser.add_argument("--weights-path", required=True,
                    help="path to weight file for autoencoder.")
parser.add_argument("--n-clusters", default=20, type=int,
                    help="number of clusters for kmeans.")
args = parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    itemlist = get_items(Path(args.dataset),
                         extensions=['.png'],
                         include=[f'CF_Normacolor_0{i}' for i in (234, 300, 303, 230, 182)])
    autoencoder = make_autoencoder_model(
        width=args.width, depth=args.depth, patch_size=args.patch_size)
    autoencoder.load_weights(args.weights_path)
    extractor = Model(inputs=autoencoder.input,
                      outputs=autoencoder.get_layer('encoding').output)

    outdir = Path(args.outdir)
    exp_id = getNextFilePath(outdir, "kmeans")
    kmeans = get_centroids(itemlist, extractor, outdir/f'kmeans_{exp_id}.p', n_clusters=args.n_clusters,
                           batch_size=args.batch_size, patch_size=args.patch_size)
    hist = get_histograms(itemlist, extractor, kmeans, outdir/f'hist_{exp_id}.npy',
                          batch_size=args.batch_size, patch_size=args.patch_size, img_size=args.img_size)
