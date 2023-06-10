from cleanfid import fid


def make_custom_stats(custom_name, real_fdir, mode="clean"):
    """
    generating custom statistics(save to local cache)
    will save two stats as FID and KID
    """
    fid.make_custom_stats(name=custom_name, fdir=real_fdir, mode=mode)

def compute_fid_kid(fake_fdir, custom_name, mode="clean", dataset_split="custom"):
    """
    using the generated custom statistics
    """
    score_fid = fid.compute_fid(fake_fdir, dataset_name=custom_name, mode=mode, dataset_split=dataset_split)
    score_kid_10e3 = fid.compute_kid(fake_fdir, dataset_name=custom_name, mode=mode, dataset_split=dataset_split) * 10e3
    print(f"score FID : {score_fid:.2f}, score KID 10^3 : {score_kid_10e3:.2f}")
    return score_fid, score_kid_10e3

def remove_custom_stats(custom_name, mode="clean"):
    """
    check if a custom statistic already exists. if exists then remove it.
    """
    # check if a custom statistic already exists
    if fid.test_stats_exists(custom_name, mode=mode):
        print(f"{custom_name} exists. Now remove it.")
        fid.remove_custom_stats(custom_name, mode=mode)
        print(f"{custom_name} has removed.")
    else:
        print(f"{custom_name} not exists. Please check it.")


if __name__ == '__main__':
    # custom_name = "neu_real"
    # real_fdir = "../dataset/NEU/NEU-50-r64/train/Cr"
    # fake_fdir = "../work_dir/generator/wgan-gp/Cr/epoch10000"
    
    # custom_name cant use UPPERCASE
    custom_name = "dragan_rs"
    real_fdir = "../dataset/NEU/NEU-50-r64/train/Cr"
    fake_fdir = "../work_dir/generator/wgan-gp/Cr/epoch10000"

    remove_custom_stats(custom_name)

    make_custom_stats(custom_name, real_fdir)
    compute_fid_kid(fake_fdir, custom_name)

    remove_custom_stats(custom_name)
