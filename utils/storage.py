import os
import torch
import shutil


class Storage(object):
    """The storage for loading and saving"""
    state = dict(net = None, optimizer = None, record_train = [], record_test = [])

    def __init__(self, config, net, optimizer, work_path, resume):
        self.__ckpt_path = os.path.join(
            work_path,
            config.ckpt_path,
            config.ckpt_name + ".pth.tar"
        )
        self.__ckpt_best = os.path.join(
            work_path,
            config.ckpt_path,
            config.ckpt_name + "_best.pth.tar"
        )
        self.best_acc, self.last_epoch = self.__load_ckpt(net, optimizer, resume)


    def __load_ckpt(self, net, optimizer, resume):
        best_acc, last_epoch = 0, 0

        if resume and os.path.isfile(self.__ckpt_path):
            self.state = torch.load(self.__ckpt_path)

            net.load_state_dict(self.state["net"], strict=False)
            optimizer.load_state_dict(self.state["optimizer"])
            try:
                best_acc = max(self.state["record_test"])
            except:
                pass
            last_epoch = len(self.state["record_train"])

        return best_acc, last_epoch


    def save_ckpt(self, net, optimizer, train_acc, test_acc=None):
        self.state["net"] = net.state_dict()
        self.state["optimizer"] = optimizer.state_dict()
        self.state["record_train"].append(train_acc)
        if test_acc is not None:
            self.state["record_test"].append(test_acc)
            torch.save(self.state, self.__ckpt_path)
            if test_acc > self.best_acc:
                shutil.copyfile(self.__ckpt_path, self.__ckpt_best)
                self.best_acc = test_acc
        else:
            torch.save(self.state, self.__ckpt_path)

        self.last_epoch += 1
