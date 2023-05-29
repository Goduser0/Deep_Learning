import os
import re
import time


def logger(config):
    if config.log:
        assert os.path.exists(config.log_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.classification_net + ' ' + config.dataset_class + ' ' + config.time
        with open(config.log_dir + '/' + filename + '.txt', 'w') as f:
            content = str(config)
            content_list = re.split(r"\(|\)", content)
            content_list = content_list[1].split(",")
            for content_line in content_list:
                key_value = content_line.split("=")
                f.write(f"{key_value[0]}:\t{key_value[1]}\n")
        f.close()
    else:
        pass
    