import os
import re
import time


##########################################################################################################
# FUNCTION: S_trainer_logger
##########################################################################################################
def S_trainer_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + ' ' + config.catagory + ' ' + config.time
        with open(config.logs_dir + '/' + filename + '.txt', 'w') as f:
            content = str(config)
            content_list = re.split(r"\(|\)", content)
            content_list = content_list[1].split(",")
            for content_line in content_list:
                key_value = content_line.split("=")
                f.write(f"{key_value[0]}:\t{key_value[1]}\n")
        f.close()
    else:
        pass
    
##########################################################################################################
# FUNCTION: S2T_trainer_logger
##########################################################################################################
def S2T_trainer_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_target + ' ' + config.catagory + ' ' + config.time
        with open(config.logs_dir + '/' + filename + '.txt', 'w') as f:
            content = str(config)
            content_list = re.split(r"\(|\)", content)
            content_list = content_list[1].split(",")
            for content_line in content_list:
                key_value = content_line.split("=")
                f.write(f"{key_value[0]}:\t{key_value[1]}\n")
        f.close()
    else:
        pass