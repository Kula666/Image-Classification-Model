import os
import matplotlib.pyplot as plt


def learning_curve(config, record_train, work_path, record_test=None):
    plt.style.use(config.learning_curve.style)
    plt.plot(range(1, len(record_train) + 1), record_train, marker='.', label="train acc")
    if record_test is not None:
        plt.plot(range(config.eval_freq, len(record_train) + \
                       config.eval_freq, config.eval_freq), \
                 record_test, marker='.', label="test acc")

    plt.legend(loc=4)
    plt.title("{} learning curve ({})".\
              format(config.architecture, config.data_name))
    plt.xticks(range(0, len(record_train) + 1, \
                     config.learning_curve.xtick_step))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    if config.learning_curve.save_path is not None:
        save_path = os.path.join(work_path,
                        config.learning_curve.save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + "lr_curve.jpg")
    plt.show()
