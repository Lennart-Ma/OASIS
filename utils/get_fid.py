"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
from scipy import linalg
# from scipy.misc import imread
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from utils.inception import InceptionV3
import utils.metric_utils

import models.models as models


def calc_mean_std_of_batch(batch):
    
    mean = 0.0
    std = 0.0
    for i in range(len(batch)):
        mean += torch.mean(batch[i], dim=(1, 2))
        std += torch.std(batch[i], dim=(1, 2))
    
    return mean/len(batch), std/len(batch)

def normalize_batch_to_values_between_0_1(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def get_activations(fakes, dataloader, n_imgs, img_size, model, batch_size, g_model, opt, dims=2048,
                    cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    n_batches = n_imgs // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    if len(img_size) != 2:
        raise ValueError('Only 2D images are supported for FID, but image size has {0} dimensions.'.format(len(img_size)))

    for i, data_i in tqdm(enumerate(dataloader)):
        
        start = i * batch_size
        end = start + batch_size
        image, label = models.preprocess_input(opt, data_i)
        if fakes:
            augmented_label_maps = utils.metric_utils.augment_label_map(label)
            with torch.no_grad():
                # generated_aug = utils.metric_utils.generate_image_from_seg_mask(augmented_label_maps, netG, opt)
                # generated_no_aug = utils.metric_utils.generate_image_from_seg_mask(label_maps_no_aug, netG, opt)
                generated_no_aug = g_model(image, label, "generate")
            batch = generated_no_aug
            batch = normalize_batch_to_values_between_0_1(batch)
                
        else:
            batch = image
            batch = normalize_batch_to_values_between_0_1(batch)

        if img_size != list(batch.shape[-2:]):
            print("inside batch interpolate")
            batch = torch.nn.functional.interpolate(batch, size=img_size)

        batch = batch.expand(-1, 3, -1, -1)  # grayvalue image  to rgb
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            print("apply average pooling function is called")
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr

def calculate_activation_statistics(fakes, dataloader, n_imgs, img_size, model, batch_size, g_model, opt,
                                    dims=2048, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(fakes, dataloader, n_imgs, img_size, model, batch_size, g_model, opt, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_dataloader(dataloader, n_imgs, img_size, batch_size, g_model, opt, dims, cuda):
    """Calculates the FID
    fakes: if True generates fake images from augmented segmentation masks taken from dataloader data_i['label']
    
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    print('Loading Inception-v3 ...')
    model = InceptionV3([block_idx])
    print('Loading Inception-v3 finished')
    if cuda:
        model.cuda()
        model.eval()

    m1, s1 = calculate_activation_statistics(False, dataloader, n_imgs, img_size, model, batch_size, g_model, opt,
                                         dims, cuda)

    m2, s2 = calculate_activation_statistics(True, dataloader, n_imgs, img_size, model, batch_size, g_model, opt,
                                         dims, cuda)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
