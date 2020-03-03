import os
import yaml
import argparse
import torch.nn as nn
from utils import *
from model import *
from easydict import EasyDict as edict


parse = argparse.ArgumentParser(description="=== PyTorch Image Classification ===")
parse.add_argument("-p", "--work_path", required=True, help="path to work")
parse.add_argument("-r", "--resume", action="store_true", help="resume from checkpoint")
args = parse.parse_args()


with open(os.path.join(args.work_path, "config.yaml")) as f:
    config = edict(yaml.load(f, yaml.FullLoader))


def main():
    net = get_model(config)
    optimizer = get_optimizer(config, net)
    criterion = nn.CrossEntropyLoss()

    transform_train, transform_test = get_transforms(config), get_transforms(config, True)
    train_loader, test_loader = load_data(config, transform_train, transform_test)
    storage = Storage(config, net, optimizer, args.work_path, args.resume)
    trainer = Trainer(config, net, optimizer, criterion)
    tester = Tester(config, net, criterion)

    record_train = storage.state["record_train"]
    record_test = storage.state["record_test"]

    logger = Logger(config, args.work_path).get_log()

    logger.debug(trainer)

    for epoch in range(storage.last_epoch + 1, config.num_epochs + 1):
        logger.debug("========== epoch: [{}/{}] ==========".format(epoch, config.num_epochs))
        train_loss, train_acc = trainer.train(train_loader, logger)
        test_loss, test_acc = None, None

        if epoch % config.eval_freq == 0:
            test_loss, test_acc = tester.test(test_loader, logger)

        storage.save_ckpt(net, optimizer, train_acc, test_acc)
        if epoch % config.learning_curve.draw_freq == 0:
            learning_curve(config, record_train, args.work_path, record_test)
        lr_schedule(config, epoch, optimizer)

    logger.warning("Training Finished ==> best accuracy: {:6.3f}%".format(storage.best_acc))
    learning_curve(config, record_train, args.work_path, record_test)


if __name__ == '__main__':
    main()
