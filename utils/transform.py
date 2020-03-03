import torchvision.transforms as transforms


__all__ = ["get_transforms"]


def get_transforms(config, is_train=True):
    aug = list()
    if is_train:
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, 4))
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())
    aug.append(transforms.ToTensor())
    if config.augmentation.normalize:
        if config.data_name == 'CIFAR10':
            aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        else:
            aug.append(transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
    return transforms.Compose(aug)
