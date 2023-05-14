import time
import os

def logger(config):
    if config.log:
        assert os.path.exists(config.log_dir), 'Error: No config.log_dir'
        filename = config.net + config.dataset_class + ' ' + time.strftime("%Y-%m-%d__%H-%M-%S", time.localtime())
        with open(config.log_dir + '/' + filename + '.txt', 'w') as f:
            f.write(str(config))
        f.close()
    else:
        pass