# MHA-JAM

## Environment Installation
1- clone the repository 

2- Using anaconda you can create an envirnment as follow: 
`conda env create -f environment.yml`

3- Alternatively, you can first create an environment using conda or virtual env and then install requirements as follow: 
```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

## Data preparation
1- Download and extract the Full dataset v1.0 [NuScenes dataset](https://www.nuscenes.org/download?externalData=all&mapData=all&modalities=Any) and the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit). 
Notes: 
- An account is needed to download the files
- The dataset is big, so mini data can be used for quick experimenting
- You will need to download the latest 'Map expansion' and extract that inside the maps folder.

2- Generate the files needed for training from the dataset:
```
python data_generator.py --mode="v1.0-mini" --dataset_dir='/home/bassel/repos/nuscenes/v1.0-mini' --out_dir="/home/bassel/repos/nuscenes/mha-jam"
```
Parameters: 
* `mode` is used to switch between `v1.0-mini` and `v1.0-trainval` versions (mini or big set). 
* `dataset_dir` accepts the path of the dataset
* `out_dir` accepts the output dir for generating the training files (which are needed for training in the next step)


3- Start training:
```
python train_model.py --mode="v1.0-mini" --preprocessed_dataset_dir="/home/bassel/repos/nuscenes/mha-jam" --model_type="JAM" --model_save_dir "./model_iterations" --save_best_model_only=True --epochs=500 --batch_size=8 
```
Parameters
 * `mode` is used to switch between `v1.0-mini` and `v1.0-trainval` versions (mini or big set). 
 * `preprocessed_dataset_dir` is set to the path of the training files (generated in the previous step).
 * `model_type` is set to either `JAM` or `SAM` to switch between the 2 models types
 * `model_save_dir` accepts the path of the dir to save the model
 * `save_best_model_only` is set to `True` or `False` to overwrite the best model in the same file vs saving each model decreasing the loss
 * `epochs` controls the number of epochs as desired
 * `batch_size` controls the size of the batch
 

4- Evaluate the model:
```
python evaluate_model.py
```
Parameters
 *`mode` is used to switch between `v1.0-mini` and `v1.0-trainval` versions (mini or big set).
 *`preprocessed_dataset_dir` is set to the path of the training files (generated in step `#2`).
 *`model_path` this is set to either a dir containing several models (at least 1 model) or the path to a specific model file. \
For a dir, each model existing is being evaluated nad the minimum errors are reported. For a single file, the passed model error is reported.  

