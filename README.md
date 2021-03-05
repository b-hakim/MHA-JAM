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
1- Download the [NuScenes dataset](https://www.nuscenes.org/download?externalData=all&mapData=all&modalities=Any) and the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

2- Extract all the files

3- Edit the path for the dataset and the output path in the `data_generator.py` and run it.
It is recommended to use the mini version first. To do so, set the mode to "v1.0-mini" before runing the script `data_generator.py`

4- In `train_model.py`:
 - Set the `preprocessed_dataset_dir` to the output path set in step `#3`.
 - Modify the number of epochs as desired
 - Set the `model_type` to either `MODEL_TYPE.MHA_JAM` or `MODEL_TYPE.MHA_SAM`
 - Run the script
 # ToDo: Pass a path to save the model into. Currently, the model is assumed to lie in model_iterations 
 # ToDo: Pass a flag to save model after each iteration, save best model only, or latest model
 
5- In `evaluate_model.py` Set the `preprocessed_dataset_dir` to the output path set in step `#3` and run it to get the best model.
#ToDo: Allow model path to be passed as an input. Currently, the model is assumed to lie in model_iterations
