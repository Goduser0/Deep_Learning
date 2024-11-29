# conda create -n dzw_gan python=3.7
# conda activate dzw_gan

# install pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchsummary==1.5.1
pip install torchinfo==1.6.5

# install tensorflow
conda install cudatoolkit==11.3.1
conda install cudnn==8.2.1
pip install tensorflow-gpu==2.5.0

# others packages
pip install tqdm==4.63.0
pip install scikit-image==0.18.3
pip install numpy==1.21.5
pip install opencv-python==4.5.5.64
pip install deformable-attention
pip install scikit-learn
pip install pot
pip install imgaug
pip install scikit-learn==0.24.2

# install caculate fid
pip install pytorch-fid
pip install clean-fid

# install timm
cd /home/user/duzongwei/Projects/FSGAN/third_party/Timm/pytorch-image-models && pip install -e .
