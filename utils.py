import torch
import numpy as np
from models import *
from PIL import Image, ExifTags
import os
import cv2
import inspect
import operator
from pathlib import Path


# Only complete, named architecture factories are allowed. Do not evaluate CLI input.
MODEL_FACTORIES = {
    name: globals()[name] for name in (
        'DPN26', 'DPN92', 'SENet18', 'DenseNet105', 'DenseNet121',
        'DenseNet169', 'DenseNet201', 'DenseNet161', 'densenet_cifar',
        'GoogLeNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
        'ResNet152', 'ResNeXt29_2x64d', 'ResNeXt29_4x64d',
        'ResNeXt29_8x64d', 'ResNeXt29_32x4d', 'MobileNetV2',
    )
}


class _CPUDataParallel(torch.nn.DataParallel):
    """Keep the legacy .module API without selecting available CUDA devices."""

    def __init__(self, module):
        # DataParallel.forward directly calls .module when device_ids is empty.
        # Its normal constructor discovers CUDA even for an explicitly CPU model.
        torch.nn.Module.__init__(self)
        self.module = module
        self.device_ids = []


def _nonnegative_index(value, name):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if result < 0:
        raise ValueError(f'{name} must be nonnegative')
    return result


def _load_split(path, subset):
    # Split files contain PIL images and must be trusted local project artifacts.
    return torch.load(path, map_location='cpu', weights_only=False)[subset]


def load_pretrained_net(net_name, chk_name, model_chk_path, test_dp=0, bdp=0, device='cuda'):
    """Load a trusted local checkpoint on an explicit device; retain the .module API."""
    if net_name not in MODEL_FACTORIES:
        raise ValueError(f'Unsupported model {net_name!r}; choose from {sorted(MODEL_FACTORIES)}')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use device="cpu"')
    factory = MODEL_FACTORIES[net_name]
    parameters = inspect.signature(factory).parameters
    options = {}
    for name, value in [('test_dp', test_dp), ('bdp', bdp)]:
        if name in parameters:
            options[name] = value
        elif value != 0:
            raise ValueError(f'{net_name} does not support {name}={value}')
    model = factory(**options)
    filename = str(chk_name) % net_name if '%s' in str(chk_name) else str(chk_name)
    checkpoint_path = Path(model_chk_path) / filename
    print('==> Resuming from checkpoint for %s..' % net_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Checkpoint must contain a state dict: {checkpoint_path}')
    state = checkpoint.get('net', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Checkpoint does not contain a nonempty state dict: {checkpoint_path}')
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    # Historical trainers use net.module, including on CPU. For cuda:N the primary
    # DataParallel device must match the device holding the model parameters.
    if device.type == 'cuda':
        primary = device.index if device.index is not None else torch.cuda.current_device()
        net = torch.nn.DataParallel(model, device_ids=[primary], output_device=primary)
    else:
        net = _CPUDataParallel(model)
    net.eval()
    return net


def crop_resize_opencv(path):
    try:
        img = Image.open(path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    w, h = img.size

    img = img.crop((0, 0, min([w, h]), min([w, h])))
    w, h = img.size

    if w > h:
        l = (w - h) / 2
        img = img.crop((l, 0, w - l, h))
    w, h = img.size
    assert w == h, "size error, {}, {}!".format(w, h)

    imgnp = np.asarray(img)
    img.close()
    imgcv2 = cv2.resize(imgnp, (32, 32), interpolation=cv2.INTER_AREA)

    return imgcv2


def fetch_camera_target(target_index, target_print_size, transforms):
    import cv2
    path = "targets/camera/printed-size-{}/target{}.png".format(target_print_size, "%.3d" % target_index)
    if os.path.exists(path):
        img = crop_resize_opencv(path)
        img = img[:, :, :3]  # remove the alpha/transparency channel
        if transforms is not None:
            img = transforms(img)[None, :, :, :]
        else:
            img = np.array(img)[None, :, :, :]
        return img
    else:
        print("camera image for target {} does not exist".format(target_index))
        print(path)
        # assert False
        return None


def fetch_target_102flower_dset(target_index, transforms, path='datasets/102flowers'):
    target_path = '%s/image_%05d.jpg' % (path, target_index)
    assert os.path.exists(target_path)
    img = crop_resize_opencv(target_path)
    img = img[:, :, :3]
    img = transforms(img)[None, :, :, :]
    return img


def fetch_target(target_label, target_index, start_idx, path, subset, transforms):
    """
    Fetch the "target_index"-th target, counting starts from start_idx
    """
    target_index = _nonnegative_index(target_index, 'target_index')
    start_idx = _nonnegative_index(start_idx, 'start_idx')
    img_label_list = _load_split(path, subset)
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter == (target_index + start_idx + 1):
                if transforms is not None:
                    return transforms(img)[None, :, :, :]
                else:
                    return np.array(img)[None, :, :, :]
    raise ValueError(f'Target index {target_index} with start_idx={start_idx} exceeds '
                     f'{counter} samples of class {target_label} in {subset}')


def fetch_all_target_cls(target_label, num_per_class_transfer, subset, path, transforms, start_idx=0):
    """Return a contiguous per-class candidate window and its split-relative indices."""
    num_per_class_transfer = _nonnegative_index(num_per_class_transfer, 'num_per_class_transfer')
    start_idx = _nonnegative_index(start_idx, 'start_idx')
    if num_per_class_transfer == 0:
        raise ValueError('num_per_class_transfer must be positive')
    img_label_list = _load_split(path, subset)
    counter = 0
    targetcls_img_list = []
    idx_list = []
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if start_idx < counter <= start_idx + num_per_class_transfer:
                targetcls_img_list.append(transforms(img))
                idx_list.append(idx)
    if len(targetcls_img_list) != num_per_class_transfer:
        raise ValueError(f'Requested {num_per_class_transfer} class-{target_label} candidates '
                         f'after offset {start_idx}, but only {len(targetcls_img_list)} are available')
    return torch.stack(targetcls_img_list), idx_list


def get_target_nearest_neighbor(subs_net_list, target_cls_imgs, target_img, num_imgs, idx_list, device='cuda'):
    num_imgs = _nonnegative_index(num_imgs, 'num_imgs')
    if not subs_net_list or not 0 < num_imgs <= len(idx_list) or len(idx_list) != len(target_cls_imgs):
        raise ValueError('Nearest-base selection requires models and sufficient aligned candidates')
    target_img = target_img.to(device)
    target_cls_imgs = target_cls_imgs.to(device)
    total_dists = 0
    with torch.no_grad():
        for n_net, net in enumerate(subs_net_list):
            model = net.module if isinstance(net, torch.nn.DataParallel) else net
            target_img_feat = model.penultimate(target_img)
            target_cls_imgs_feat = model.penultimate(target_cls_imgs)
            dists = torch.sum((target_img_feat - target_cls_imgs_feat) ** 2, 1).cpu().detach().numpy()
            total_dists += dists
    if not np.isfinite(total_dists).all():
        raise ValueError('Nonfinite nearest-base distances')
    min_dist_idxes = np.argsort(total_dists, kind='stable')
    # print("Selected feature dist squares: {}".format(total_dists[min_dist_idxes[:num_imgs]]))
    return target_cls_imgs[min_dist_idxes[:num_imgs]], [idx_list[midx] for midx in min_dist_idxes[:num_imgs]]


def fetch_nearest_poison_bases(sub_net_list, target_img, num_poison, poison_label, num_per_class, subset,
                               train_data_path, transforms, start_idx=0, device='cuda'):
    imgs, idxes = fetch_all_target_cls(poison_label, num_per_class, subset, train_data_path, transforms,
                                      start_idx=start_idx)

    nn_imgs_batch, nn_idx_list = get_target_nearest_neighbor(sub_net_list, imgs, target_img, num_poison,
                                                             idxes, device=device)
    base_tensor_list = [nn_imgs_batch[n] for n in range(nn_imgs_batch.size(0))]
    print("Selected nearest neighbors: {}".format(nn_idx_list))
    return base_tensor_list, nn_idx_list


def fetch_poison_bases(poison_label, num_poison, subset, path, transforms, start_idx=0):
    """
    Fetch num_poison images after a per-class offset; return split-relative indices.
    """
    num_poison = _nonnegative_index(num_poison, 'num_poison')
    start_idx = _nonnegative_index(start_idx, 'start_idx')
    if num_poison == 0:
        raise ValueError('num_poison must be positive')
    img_label_list = _load_split(path, subset)
    base_tensor_list, base_idx_list = [], []
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == poison_label:
            if counter >= start_idx:
                base_tensor_list.append(transforms(img))
                base_idx_list.append(idx)
            counter += 1
        if len(base_tensor_list) == num_poison:
            break
    if len(base_tensor_list) != num_poison:
        raise ValueError(f'Requested {num_poison} class-{poison_label} bases after offset '
                         f'{start_idx}, but only {len(base_tensor_list)} are available')
    return base_tensor_list, base_idx_list


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in
                    range(poison_batch.poison.size(0))]

    return poison_tuple


def get_poison_list(poison_batch):
    """
    Doesn't have the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().clone()) for num_p in range(poison_batch.poison.size(0))]

    return poison_tuple
