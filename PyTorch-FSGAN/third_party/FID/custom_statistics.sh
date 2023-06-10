#################################
# calculate clean FID and KID
#################################
custom_name="neu_real"
real_fdir="../dataset/NEU/NEU-300-r64/Cr"
fake_fdir="../work_dir/generator/wgan-gp/Cr"

cur_dir=$(dirname $0)
echo $cur_dir

python ${cur_dir}/custom_statistics.py \
    --custom_name ${custom_name} \
    --real_fdir ${real_fdir} \
    --fake_fdir ${fake_fdir}


#################################
# calculate pytorch FID
#################################
# You can select the dimensionality of features to use with the flag --dims N, where N is the dimensionality of features. The choices are:
# 64: first max pooling features
# 192: second max pooling features
# 768: pre-aux classifier features
# 2048: final average pooling features (this is the default)

for layer_dir in 2048 768 192 64;
do
python -m pytorch_fid ${real_fdir} ${fake_fdir}\
    --dims ${layer_dir}
done

##################################
# save as log, you can use followings
# bash ./custom_statistics.sh 2>&1 | tee mylog.log
##################################