import torch.optim as optim


__all__ = ["get_optimizer"]


def get_optimizer(config, net):
    optimizer = getattr(optim, config.optimizer.name)
    return optimizer(net.parameters(), \
                     **config.optimizer.params)
