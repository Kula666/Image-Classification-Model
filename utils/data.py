import torchvision
from torch.utils.data import DataLoader


__all__ = ["load_data"]


def load_data(config, transform_train, transform_test):
    assert hasattr(torchvision.datasets, config.data_name), \
        "config.data_name must be in the torchvision.datasets!"
    dataset = getattr(torchvision.datasets, config.data_name)
    train_set = dataset(
        root=config.train_set_path, train=True,
        download=True, transform=transform_train
    )
    test_set = dataset(
        root=config.test_set_path, train=False,
        download=True, transform=transform_test
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    return train_loader, test_loader
