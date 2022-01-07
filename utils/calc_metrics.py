from scipy.ndimage.interpolation import rotate
import torch
import os
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
import scipy.linalg
from torch.utils.data import dataset

import models.models as models
from utils import util_borrowed
from utils import metric_utils
from utils import get_fid


def compute_fid(opt, netG, dataloader, dataset_length, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opt=opt, netG=netG, dataloader=dataloader, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    print("Collected feature stats for generator")

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        dataloader, dataset_length=dataset_length, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True).get_mean_cov()

    print("Collected feature stats for dataset")

    # if opts.rank != 0:
    #     return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

def get_ssim(reals, fakes):

    # from tensor shape (B x C x H x W) to numpy array shape (B x H x W x C) --> needed for ssim()
    reals = util_borrowed.tensor2im(reals)
    fakes = util_borrowed.tensor2im(fakes)

    assert len(reals) == len(fakes)

    ssim_array = np.empty(shape=(len(reals), ), dtype=np.float32)
    for i, (real, fake) in enumerate(zip(reals, fakes)):
        ssim_array[i] = ssim(real, fake, multichannel=True)
    print('' * 100, end='\r')

    ssim_mean = np.mean(ssim_array, axis=0)
    ssim_std = np.std(ssim_array, axis=0)

    #show(fakes, "fake_test2.png")
    #show(reals, "real_test2.png")

    return ssim_mean, ssim_std

def get_JS_divergence(reals, fakes):

    """
    reals and fakes need to be on cpu
    """

    reals = reals.flatten().cpu()
    fakes = fakes.flatten().cpu()

    hist_reals, _ = np.histogram(reals, bins=255)
    hist_fakes, _ = np.histogram(fakes, bins=255)

    return(distance.jensenshannon(hist_reals, hist_fakes))

def plot_hist(reals, fakes, run_dir, epoch):
    """
    Plots four histograms of the pixel value distribution of two arrays of images (overlapping) and saves it to path
    First histogram with all 255 values, second with all values on log scale, the third histogram with value 0 cut out
    and the fourth histogram with value 0 cut out and on log scale
    
    PARAMETERS
    
    reals: array like
    array of real images in the form (B x C x W x H)

    fakes: array like
    array of fakes images in the form (B x C x W x H)

    run_dir: string
    path to the location where the histogram should be saved
    """

    reals_flat = reals.flatten().cpu()
    fakes_flat = fakes.flatten().cpu()

    _ = plt.hist(reals_flat, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution all values")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_{epoch}_full_.png'))
    plt.close()

    _ = plt.hist(reals_flat, log=True, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat, log=True, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution all values log")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_{epoch}_log_.png'))
    plt.close()


    ## Doesnt work somehow
    reals_flat_del = np.delete(reals_flat, np.where(reals_flat == 0))
    
    fakes_flat_del = np.delete(fakes_flat, np.where(fakes_flat == 0))

    _ = plt.hist(reals_flat_del, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat_del, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution w/o value=0")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_{epoch}_wo0_.png'))
    plt.close()

    _ = plt.hist(reals_flat_del, log=True, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat_del, log=True, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution w/o value=0 log")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_{epoch}_wo0_log.png'))
    plt.close()

def save_generated_to_png(generated, save_path, n_imgs):

    """
    Saves n_imgs of generated fake images or augmented segmentation maks to save_path

    PARAMETERS:
    generated: torch.tensor
    can either be the model output (fake image) of shape (B x C=3 x H x W) where channel is RGB and not n_classes
    or the augmented segmentation map used to generate a fake image of shape (B x 1 x H x W)

    save_path: string
    path w/o name of file

    n_imgs: int
    how many images should be saved
    """
    if generated.shape[1] == 3:
        images_numpy = util_borrowed.tensor2im(generated)
        for i in range(n_imgs):
            image_numpy = images_numpy[i]
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(os.path.join(save_path, f"aug_img_{i}.png"))


    elif generated.shape[1] == 1:
        # this gets called if we pass a segmentation mask as generated
        images_numpy = generated.numpy().transpose(0,2,3,1).astype(dtype=np.uint8) 
        for i in range(n_imgs):
            image_numpy = images_numpy[i]
            image_numpy[image_numpy==3] = 255
            image_numpy[image_numpy==2] = 150
            image_numpy[image_numpy==1] = 50
            image_numpy = np.squeeze(image_numpy, axis=2)
            image_pil = Image.fromarray(image_numpy).convert("L")
            image_pil.save(os.path.join(save_path, f"aug_mask_{i}.png"))


def get_metrics(opt, model, dataloader, epoch):


    reals_list = []
    generated_aug_list = []
    generated_no_aug_list = []
    augmented_label_maps_list = []
    for _, data_i in enumerate(dataloader):

        image, label = models.preprocess_input(opt, data_i)
        
        label_maps_no_aug = label
        augmented_label_maps = metric_utils.augment_label_map(label)

        with torch.no_grad():
            generated_aug = model(image, augmented_label_maps, "generate")
            generated_no_aug = model(image, label_maps_no_aug, "generate")
        
        reals_list.append(image)
        generated_aug_list.append(generated_aug)
        generated_no_aug_list.append(generated_no_aug)
        augmented_label_maps_list.append(augmented_label_maps)

    fakes_aug = torch.cat(generated_aug_list)
    fakes_no_aug = torch.cat(generated_no_aug_list)
    reals = torch.cat(reals_list)
    aug_labels = torch.cat(augmented_label_maps_list)

    ssim_mean_aug, ssim_std_aug = get_ssim(reals, fakes_aug)
    ssim_mean_no_aug, ssim_std_no_aug = get_ssim(reals, fakes_no_aug)
    js_div_aug = get_JS_divergence(reals, fakes_aug)
    js_div_no_aug = get_JS_divergence(reals, fakes_no_aug)
    plot_hist(reals, fakes_aug, run_dir=opt.checkpoints_dir, epoch=epoch)
    
    ##############################-*
    fid = get_fid.calculate_fid_given_dataloader(dataloader, n_imgs=len(fakes_aug), img_size=[256,256], batch_size=opt.batch_size, g_model=model, opt=opt, dims=2048, cuda=True,)

    metrics = {
        "SSIM mean" : ssim_mean_no_aug,
        "SSIM std" : ssim_std_no_aug,
        "SSIM mean aug" : ssim_mean_aug,
        "SSIM std aug" : ssim_std_aug,
        "Jennson shannon divergence aug" : js_div_aug,
        "Jennson shannon divergence no aug" : js_div_no_aug,
        "FID" : fid
    }

    for key, value in metrics.items():
        print(key, value)

    return metrics