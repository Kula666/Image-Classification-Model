import os
import logging
import colorlog


class Logger(object):
    log_map = dict(train = "DEBUG", cost_time = "INFO", test = "WARNING")

    def __init__(self, config, work_path):
        log_colors = {
            self.log_map["train"]: config.log_color.train,
            self.log_map["cost_time"]: config.log_color.cost_time,
            self.log_map["test"]: config.log_color.test
        }
        self.__logger = colorlog.getLogger(config.log_name)
        self.__logger.setLevel(logging.DEBUG)

        log_path = os.path.join(work_path, config.log_path, "log.txt")
        file_handler = logging.FileHandler(log_path)
        console_handler = colorlog.StreamHandler()

        color_formatter = colorlog.ColoredFormatter(
                    "%({})s[%(asctime)s] - %(log_color)s%(message)s" \
                        .format(config.log_color.time), log_colors=log_colors)

        file_formatter = logging.Formatter(
            '[%(asctime)s] - %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(color_formatter)

        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)


    def get_log(self):
        return self.__logger

