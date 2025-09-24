from torchvision.transforms.functional import affine
from scipy.ndimage import shift
from skimage.util import random_noise
import numpy as np
import random
from torch import tensor

def apply_augmentations(data, aug_types, random_choise=False, learn=False):
    augmentated_data = data.copy()
    if random_choise:
        aug_types = [np.random.choise(aug_types + [None])]
        if aug_types[0] is None: aug_types = []

    for aug_type in aug_types:
        if aug_type == 'rotate':
            for i, image in enumerate(augmentated_data):
                rotated_image = affine(tensor(image).reshape((1, 28, 28)), angle=random.random() * 30 - 15, translate=(0, 0), scale=1.0, shear=0, fill=-0.42421296)
                augmentated_data[i] = rotated_image if not learn else rotated_image.reshape((784))
        elif aug_type == 'shift':
            for i, image in enumerate(augmentated_data):
                shift_x = np.random.uniform(-3, 3)
                shift_y = np.random.uniform(-3, 3)
                shifted_image = shift(image, [0, shift_x, shift_y], mode='constant', cval=-0.42421296)
                augmentated_data[i] = shifted_image
        elif aug_type == 'noise':
            for i, image in enumerate(augmentated_data):
                noised_image = random_noise(image, mode='gaussian', var=0.005)
                augmentated_data[i] = noised_image
    return augmentated_data