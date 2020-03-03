import time
import torch.optim as optim
from .scheduler import get_cur_lr


class Trainer(object):
    def __init__(self, config, net, optimizer, criterion):
        self.__config = config
        self.__net = net
        self.__optimizer = optimizer
        self.__criterion = criterion


    def train(self, train_loader, logger):
        self.__net.train()

        start = time.time()
        total, correct, train_loss = 0, 0, 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(self.__config.device), y.to(self.__config.device)
            output = self.__net(X)
            loss = self.__criterion(output, y)

            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_loss += loss.item()
            train_acc = correct / total * 100

            if (i + 1) % self.__config.num_print == 0:
                logger.debug("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                      .format(i + 1, len(train_loader), \
                              train_loss / (i + 1), train_acc, get_cur_lr(self.__optimizer)))

        logger.info("--- cost time: {:.4f}s ---".format(time.time() - start))
        return train_loss / len(train_loader), train_acc


    def __count_params(self):
        """Calculate the number of parameters"""
        return sum(p.numel() for p in self.__net.parameters() if p.requires_grad)


    def __str__(self):
        return str(self.__config) + "\n" + str(self.__net) + "\n" + "-" * 20 + \
               " Training parameters: {} ".format(self.__count_params()) + "-" * 20
