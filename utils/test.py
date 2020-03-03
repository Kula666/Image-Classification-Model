import torch


class Tester(object):
    def __init__(self, config, net, criterion):
        self.__config = config
        self.__net = net
        self.__criterion = criterion


    def test(self, test_loader, logger):
        self.__net.eval()
        total, correct, test_loss = 0, 0, 0

        with torch.no_grad():
            logger.warning("*************** test ***************")
            for X, y in test_loader:
                X, y = X.to(self.__config.device), y.to(self.__config.device)

                output = self.__net(X)
                loss = self.__criterion(output, y)

                total += y.size(0)
                correct += (output.argmax(dim=1) == y).sum().item()
                test_loss += loss.item()

        test_acc = correct / total * 100
        test_loss /= len(test_loader)

        logger.warning("test_loss: {:.3f} | test_acc: {:6.3f}%" \
              .format(test_loss, test_acc))
        logger.warning("************************************\n")

        return test_loss, test_acc
