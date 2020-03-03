def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def lr_schedule(config, epoch, optimizer):
    lr = get_cur_lr(optimizer)

    if config.lr_scheduler.type == "STEP":
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
