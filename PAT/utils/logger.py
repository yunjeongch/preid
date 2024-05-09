import logging
import os
import sys
import os.path as osp
import datetime
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "{:%m%d}_{:%H:%M:%S}_train.log".format(datetime.datetime.now(), datetime.datetime.now())), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "{:%m%d}_{:%H:%M:%S}_test.log".format(datetime.datetime.now(), datetime.datetime.now())), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger