from cleanfid import fid
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_name", type=str, default="neu_real", help="custom dataset name")
    parser.add_argument("--real_fdir", type=str, default="/home/user/duzongwei/Packages/FID/sample_test/neu_real/cr", help="real samples dir path")
    parser.add_argument("--fake_fdir", default="/home/user/duzongwei/Packages/FID/sample_test/neu_fake/cr", help="generator samples dir path")
    args = parser.parse_args()
    return args

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
    # args
    args = parse_args()
    custom_name = args.custom_name
    real_fdir = args.real_fdir
    fake_fdir = args.fake_fdir
    
    # test 
    # custom_name = "neu_real"
    # real_fdir = "/home/user/duzongwei/Packages/FID/sample_test/neu_real/cr"
    # fake_fdir = "/home/user/duzongwei/Packages/FID/sample_test/neu_fake/cr"
    
    remove_custom_stats(custom_name)
    
    make_custom_stats(custom_name, real_fdir)
    compute_fid_kid(fake_fdir, custom_name)
    