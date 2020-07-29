import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb, skimage

import sys, glob
import numpy as np
from PIL import Image
import ipdb
# import shutil
import time
import os
# from utils import mkdir_p

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
# import imageio
from termcolor import colored
# from naman_robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT

import utils as eutils

import warnings
warnings.filterwarnings("ignore")

import argparse
import settings

use_cuda = torch.cuda.is_available()
## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path to the input image dir', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )

    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Pre (1)', default=1,
                        )

    parser.add_argument('-n_mean', '--noise_mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-n_var', '--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    parser.add_argument('-n_seed', '--noise_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-if_n', '--if_noise', type=int, choices=range(2),
                        help='Whether to add noise to the image or not. Default: 0', default=0,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 1735', default=1735,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size for the batch of images. Default: 1', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.noise_seed is not None:
        print(f'Setting the numpy seed with value: {args.noise_seed}')
        np.random.seed(args.noise_seed)

    if args.img_dir_path is None:
        print('Please provide path to image dir. Exiting')
        sys.exit(1)
    else:
        args.img_dir_path = os.path.abspath(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    args.batch_size = 1 ## to make sure only 1 image is being ran. you can chnage it if you like

    return args


if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    im_label_map = eutils.imagenet_label_mappings()

    if args.if_pre == 1:
        softmax = 'pre'
    else:
        softmax = 'post'

    ############################################
    ## #Indices for images
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             ])

    pytorch_data_loader, img_count = eutils.load_data(args.img_dir_path, pytorch_preprocessFn,
                                                      img_idxs=[args.start_idx, args.end_idx],
                                                      batch_size=args.batch_size,
                                                      idx_flag=args.idx_flag, args=args)
    madry_data_loader, img_count = eutils.load_data(args.img_dir_path, madry_preprocessFn,
                                                    img_idxs=[args.start_idx, args.end_idx],
                                                    batch_size=args.batch_size,
                                                    idx_flag=args.idx_flag, args=args)

    # ############################
    # ## # Creating Noise
    # if args.if_noise == 1:
    #     noise = torch.from_numpy(np.random.normal(args.noise_mean,
    #                                               args.noise_var ** 0.5,
    #                                               (3, 224, 224))).float().unsqueeze(0)
    #     if use_cuda:
    #         noise = noise.cuda()

    ############################
    model_names = []
    model_names.append('madry') # ResNet-R
    model_names.append('pytorch') # ResNet
    model_names.append('googlenet') #GoogleNet
    model_names.append('madry_googlenet')  #GoogleNet-R


    data_loader_dict = {'pytorch': pytorch_data_loader,
                        'madry': madry_data_loader,
                        'madry_googlenet': madry_data_loader,
                        'googlenet': pytorch_data_loader}
    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}
    im_sz_dict = {'pytorch': 224,
                  'madry': 224,
                  'madry_googlenet': 224,
                  'googlenet': 224}
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}

    ############################
    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        data_loader = data_loader_dict[model_name]
        im_sz = im_sz_dict[model_name]

        if args.batch_size > 1:
            out_dir = os.path.join(args.out_path, f'Grad_{model_name}')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print(f'Saving results in {out_dir}')

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=args.if_pre)  # Returns logits

        par_name = f'softmax_{softmax}_idx_flag_{args.idx_flag}_start_idx_{args.start_idx}_' \
                   f'end_idx_{args.end_idx}_if_noise_{args.if_noise}_' \
                   f'seed_{args.noise_seed}_mean_{args.noise_mean}_' \
                   f'var_{args.noise_var}_model_name_{model_name}'

        for i, (img, targ_class, img_path) in enumerate(data_loader):
            batch_time = time.time()
            model.zero_grad()

            if args.batch_size == 1:
                ## only for batch size of 1
                img_name = img_path[0].split('/')[-1].split('.')[0]
                out_dir = os.path.join(args.out_path, img_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                print(f'Saving results in {out_dir}')

            print(f'Analysing batch: {i} of size {len(targ_class)}')
            targ_class = targ_class.cpu()
            sz = len(targ_class)

            if use_cuda:
                img = img.cuda()

            ## #We want to compute gradients
            img = Variable(img, requires_grad=True)


            ## #Prob and gradients
            sel_nodes_shape = targ_class.shape
            ones = torch.ones(sel_nodes_shape)
            if use_cuda:
                ones = ones.cuda()

            if args.if_pre == 1:
                print('Pre softmax analysis')
                logits = model(img)
                probs = F.softmax(logits, dim=1).cpu()
                sel_nodes = logits[torch.arange(len(targ_class)), targ_class]
                sel_nodes.backward(ones)
                logits = logits.cpu()

            else:
                print('Post softmax analysis')
                probs = model(img)
                sel_nodes = probs[torch.arange(len(targ_class)), targ_class]
                sel_nodes.backward(ones)
                probs = probs.cpu()

            grad = img.grad.cpu().numpy() #[2, 3, 224, 224]
            grad = np.rollaxis(grad, 1, 4) #[2, 224, 224, 3]
            grad = np.mean(grad, axis=-1)

            # if i > 0:
            #     ipdb.set_trace()

            img_path = np.asarray(list(img_path), dtype=str)

            if args.batch_size == 1:
                ## only for batch size of 1
                np.save(os.path.join(out_dir, f'grad_{par_name}.npy'), grad[0])
            else:
                np.savetxt(os.path.join(out_dir, f'time_{f_time}_img_paths_{par_name}_batch_idx_{i:02d}_batch_size_{sz:04d}.txt'), img_path, fmt='%s')
                np.save(os.path.join(out_dir, f'time_{f_time}_heatmaps_{par_name}_batch_idx_{i:02d}_batch_size_{sz:04d}.npy'), grad)

            print(f'Time taken for a batch is {time.time() - batch_time}\n')

    ##########################################
    print(f'Time taken is {time.time() - s_time}')
