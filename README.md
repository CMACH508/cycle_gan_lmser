# GAN-Lmser(-n) model 

by Haote Yang, SJTU


## Introduction
This project mainly introduce the work of this paper:

[G-Lmser:A GAN-Lmser Network for Image-to-Image Translation][GLMSER]

[GLMSER]:https://plesase put right address here

Some part of code is based on this [project][cyclegan]

[cyclegan]:https://github.com/leehomyc/cyclegan-1

* gan-lmser-n model:

    * main_dpn.py & model_dpn(_v2, _v3).py
    * cgl_exp_102.json with neuron_sharing: off
* gan-lmser model: 
 
    * main_dpn.py & model_dpn(_v2, _v3).py
    * cgl_exp_102.json with neuron_sharing: on  

## Environment
* tensorflow 1.8.0 

## Dataset Download

* Download a dataset (e.g. horse2zebra):
```bash
bash ./download_datasets.sh horse2zebra
```

* Create the csv file as input to the data loader. 
	* Edit the cyclegan_datasets.py file. For example, if you have a face2ramen_train dataset which contains 800 face images and 1000 ramen images both in PNG format, you can just edit the cyclegan_datasets.py as following:
	```python
	DATASET_TO_SIZES = {
    'face2ramen_train': 1000
	}

	PATH_TO_CSV = {
    'face2ramen_train': './cycle_gan_lmser/input/face2ramen/face2ramen_train.csv'
	}

	DATASET_TO_IMAGETYPE = {
    'face2ramen_train': '.png'
	}

	``` 
	* Run create_cyclegan_dataset.py:
	```bash
	python -m cycle_gan_lmser.create_cyclegan_dataset --image_path_a=folder_a --image_path_b=folder_b --dataset_name="horse2zebra_train" --do_shuffle=0
	```

## Start Training 
* Create the configuration file. The configuration file contains basic information for training/testing. An example of the configuration file could be fond at configs/exp_01.json. 

* Start training:
```bash
python -m cycle_gan_lmser.main \
    --to_train=1 \
    --log_dir=/output/cgl_model/exp_01 \
    --config_filename=cycle_gan_lmser/configs/exp_01.json
```
* Check the intermediate results.
	* Tensorboard
	```bash
	tensorboard --port=6006 --logdir=cycle_gan_lmser/output/cgl_model/exp_01/#timestamp# 
	```
	* Check the html visualization at cycle_gan_lmser/output/cgl_model/exp_01/#timestamp#/epoch_#id#.html.  

## Continue to Train
```bash
python -m cycle_gan_lmser.main \
    --to_train=2 \
    --log_dir=cycle_gan_lmser/output/cgl_model/exp_01 \
    --config_filename=cycle_gan_lmser/configs/exp_01.json \
    --checkpoint_dir=cycle_gan_lmser/output/cgl_model/exp_01/#timestamp#
```
## Test
* Create the testing dataset.
	* Edit the cyclegan_datasets.py file the same way as training.
	* Create the csv file as the input to the data loader. 
	```bash
	python -m cycle_gan_lmser.create_cyclegan_dataset --image_path_a=folder_a --image_path_b=folder_b --dataset_name="horse2zebra_test" --do_shuffle=0
	```
* Run testing.
```bash
python -m cycle_gan_lmser.main \
    --to_train=0 \
    --log_dir=cycle_gan_lmser/output/cgl_model/exp_01 \
    --config_filename=cycle_gan_lmser/configs/exp_01_test.json \
    --checkpoint_dir=cycle_gan_lmser/output/cgl_model/exp_01/#old_timestamp# 
```
The result is saved in cycle_gan_lmser/output/cgl_model/exp_01/#new_timestamp#.




