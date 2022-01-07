import torch
import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


def generate_image_from_seg_mask(image, label_map, model):

    # b = torch.bincount(torch.flatten(label_map))
    label_map[label_map == 4] = 3

    if len(label_map.size()) == 3:
        label_map = torch.reshape(label_map, (1, label_map.shape[0], label_map.shape[1], label_map.shape[2]))
        
    bs, _, h, w = label_map.size()

    nc = 4
    label_map = label_map.type(torch.LongTensor)
    input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, value=1.0)

    input_semantics = input_semantics.to(device="cuda")
    generated = model(image, label_map, "generate")

    return generated

def augment_label_map(label_maps, display=False, output_folder=None):
    """
    Possible augmentations from import imgaug.augmenters as iaa(https://github.com/aleju/imgaug):
    """

    seq = iaa.Sequential([
        iaa.Sometimes(
            0.3,
            iaa.ElasticTransformation(alpha=(0, 70.0), sigma=100.0),
        ),
        iaa.Sometimes(
            0.5,
            iaa.CropAndPad(percent=(-0.05, 0.05)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.geometric.Affine(rotate=(-15,15))
        ),
        iaa.Fliplr(0.5),
    ])

    if len(label_maps.shape) < 4:
        label_map_np = label_maps.numpy().transpose(1,2,0).astype(dtype=np.uint8)
        label_map_np_aug = seq(image=label_map_np)
    else:
        label_maps_np = label_maps.cpu().numpy().transpose(0,2,3,1).astype(dtype=np.uint8)

        label_maps_np_aug = seq(images=label_maps_np)

    if display:

        label_map_np = np.squeeze(label_map_np, axis=2)

        label_map_np_aug = np.squeeze(label_map_np_aug, axis=2)

        label_map_np[label_map_np==3] = 255
        label_map_np[label_map_np==2] = 150
        label_map_np[label_map_np==1] = 50

        label_map_np_aug[label_map_np_aug==3] = 255
        label_map_np_aug[label_map_np_aug==2] = 150
        label_map_np_aug[label_map_np_aug==1] = 50

    if output_folder is not None:
            
        label_map_pil = Image.fromarray(label_map_np).convert("L")

        label_map_pil.save(os.path.join(output_folder, "label_map.png"))

        label_map_aug_pil = Image.fromarray(label_map_np_aug).convert("L")

        label_map_aug_pil.save(os.path.join(output_folder, "label_map_aug.png")) 

    label_maps_np_aug= label_maps_np_aug.transpose(0,3,1,2)

    return torch.tensor(label_maps_np_aug)