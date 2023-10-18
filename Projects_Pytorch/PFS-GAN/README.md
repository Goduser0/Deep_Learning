# PFS-GAN
* Requirements
  * Python 2.7
  * PyTorch 1.0.1
  * TorchVision

## Input Images:
Image size: 64x128 (64x64 Source image(left) + 64x64 Target image(right))
## Training
* Baselines Training
  * **BaselineS:** ```python baselineS.py --dataset_dir <folder_path> --gpu 0```
  * **BaselineT:** ```python baselineT.py --dataset_dir <folder_path> --pretrained_model <generator_path> --gpu 0``` 
  * **CoGAN:** ```python CoGAN.py --dataset_dir_S <source_dataset_path> --dataset_dir_T <target_dataset_path> --gpu 0``` 
  * **UNIT:** ```python UNIT.py --dataset_dir_S <source_dataset_path> --dataset_dir_T <target_dataset_path> --gpu 0``` 
* PFS-GAN Training
  * **Stage1 Training:** ```python stage1.py --dataset_dir <source_dataset_path> --gpu 0 # Then, copy **'gen_#', 'enc_c_#'** into root folder.```
  * **PFS-GAN Training:** ```python PFS-GAN.py --train_dataset <training_target_dataset_path> --test_dataset <testing_dataset_path> --source_dataset <source_dataset_path> --model_name <# of model> --recon_ratio <recon_ratio> --gan_ratio <gan_ratio> --relation_ratio <relation_ratio> --gpu 0```
  
## Evaluation
* **FID:** ```python FID.py --target_folder <models_path> --base_dataset <base_dataset_path> --source_dataset <source_dataset_path> --target_dataset <training_dataset_path> --model_num <# of model> --gpu 0```
* **KID:** ```python KID.py --target_folder <models_path> --base_dataset <base_dataset_path> --source_dataset <source_dataset_path> --target_dataset <training_dataset_path> --model_num <# of model> --gpu 0```

## License for [abdulfatir/gan-metrics-pytorch](https://github.com/abdulfatir/gan-metrics-pytorch)
This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see https://arxiv.org/abs/1706.08500

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0. See https://github.com/bioinf-jku/TTUR.

