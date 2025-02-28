import os
import re

##########################################################################################################
# FUNCTION: cogan_logger
##########################################################################################################
def cogan_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_S_class + '2' + config.dataset_T_class + ' ' + config.category + ' ' + config.time
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
# FUNCTION: stage1_logger
##########################################################################################################
def stage1_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
# FUNCTION: stage2_logger
##########################################################################################################
def stage2_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
# FUNCTION: baseline_from_scratch_logger
##########################################################################################################
def baseline_from_scratch_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
# FUNCTION: baseline_finetuning_logger
##########################################################################################################
def baseline_finetuning_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + "_from_" + config.G_init_class + ' ' + config.category + ' ' + config.time
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
# FUNCTION: S_trainer_logger
##########################################################################################################
def S_trainer_logger(config):
    if config.log:
        assert os.path.exists(config.logs_dir), f"ERROR:\t({__name__}): No config.log_dir"

        filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
    filename = config.dataset_target + ' ' + config.category + ' ' + config.time
    with open(config.logs_dir + '/' + filename + '.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()

##########################################################################################################
# FUNCTION: DCGAN_logger
##########################################################################################################
def DCGAN_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()

##########################################################################################################
# FUNCTION: WGAN_GP_logger
##########################################################################################################
def WGAN_GP_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()
    
##########################################################################################################
# FUNCTION: SAGAN_logger
##########################################################################################################
def SAGAN_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()

##########################################################################################################
# FUNCTION: ConGAN_logger
##########################################################################################################
def ConGAN_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()


##########################################################################################################
# FUNCTION: CoGAN_logger
##########################################################################################################
def CoGAN_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()
    
##########################################################################################################
# FUNCTION: CycleGAN_logger
##########################################################################################################
def CycleGAN_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()

##########################################################################################################
# FUNCTION: UNIT_logger
##########################################################################################################
def UNIT_logger(config, save_dir):
    with open(save_dir + '/log.txt', 'w') as f:
        content = str(config)
        content_list = re.search(r"Namespace\((.*)\)", content).group(1)
        content_list = content_list.split(",")
        for content_line in content_list:
            key_value = content_line.split("=")
            f.write(f"{key_value[0]}:\t{key_value[1]}\n")
    f.close()