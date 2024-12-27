import os
import re

##########################################################################################################
# FUNCTION: classification_logger
##########################################################################################################
def classification_logger(config, save_dir):
    # filename = config.dataset_class + ' ' + config.classification_net + ' ' + config.time
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        # content_list = re.split(r"\(|\)", content)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()