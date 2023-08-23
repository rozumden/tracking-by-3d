import argparse
import os
import torch
import time
from my_utils import *
from tracking6d import *
from models.rendering import generate_novel_views
from segmentations import *
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="config.yaml")
    parser.add_argument("--dataset", required=False, default="concept")
    parser.add_argument("--sequence", required=False, default="09")
    parser.add_argument("--start", required=False, default=0)
    parser.add_argument("--length", required=False, default=72)
    parser.add_argument("--skip", required=False, default=1)
    parser.add_argument("--perc", required=False, default=0.15)
    parser.add_argument("--folder_name", required=False, default='public')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["image_downsample"] = args.perc
    config["tran_init"] = 2.5
    config["loss_dist_weight"] = 0

    write_folder = os.path.join(tmp_folder, args.folder_name, args.sequence)
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)
    os.makedirs(write_folder)
    os.makedirs(os.path.join(write_folder,'imgs'))
    shutil.copyfile(os.path.join('.','prototypes','model.mtl'), os.path.join(write_folder,'model.mtl'))
    config["sequence"] = args.sequence

    t0 = time.time()
    files = np.array(glob.glob(os.path.join(dataset_folder, '360photo', 'original', args.dataset, args.sequence, '*.*')))
    files.sort()
    segms = np.array(glob.glob(os.path.join(dataset_folder, '360photo', 'masks_U2Net', args.dataset, args.sequence, '*.*')))
    segms.sort()
    print('Data loading took {:.2f} seconds'.format((time.time() - t0)/1))
    if args.length is None:
        args.length = len(files)

    files = files[args.start:args.length:args.skip]
    segms = segms[args.start:args.length:args.skip]
    config["input_frames"] = len(files)
    if config["inc_step"] == 0:
        config["inc_step"] = len(files)
    print(config)

    inds = [os.path.splitext(os.path.basename(temp))[0] for temp in segms]
    baseline_dict = dict(zip(inds, segms))

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Initialize tracking')
    t0 = time.time()
    sfb = Tracking6D(config, device, write_folder, files[0], baseline_dict)
    print('Start tracking')
    best_model = sfb.run_tracking(files, baseline_dict)
    print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"], (time.time() - t0)/1, best_model["value"]))
    breakpoint()


if __name__ == "__main__":
    main()