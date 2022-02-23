## Clouds-Segmentation-Project

![AMS poster](https://github.com/DavidHuji/Clouds-Segmentation-Project/blob/master/AMS2021Poster.png)

**Project Description:**

The task is semantic segmentation of cloud satellite images. The dataset consists of IR & Visual Grayscale images of clouds
with their corresponding masks.
Each mask consists of 5 classes: Open clouds, Closed clouds, Disorganized clouds, Other clouds and Ocean (no clouds).
The main objective is to be able to separate between Open clouds, Closed clouds and no clouds - so eventually this task is a 3 class semantic segmentation task.
The available architectures are Pytorch's "DeepLabV3-ResNet101" and "Unet" and the number of output classes are 2/3/4/5.

**Some results:**

![img_15](https://user-images.githubusercontent.com/69245972/93467389-5af6fa00-f8f6-11ea-9a7b-083234ffefca.png)


![img_12](https://user-images.githubusercontent.com/69245972/93468002-00aa6900-f8f7-11ea-9035-c19232bc7ed3.png)

## Reproducing results

**Directory structure:**

In order to run the available experiments in this project, one requires a specific directory structure:
The main directory must contain:
- All the python files (which can be found in 'src' in this repository)
- The data directory (which can be found in 'data' in this repository)
- The models directory (which can be found in 'models' in this repository)
- A directory named 'weights' (will be explained shortly)
- A directory named 'results' (will be explained shortly)


**Running experiments:**

Running an experiment requires several arguments:
- Data directory: Using the directory structure described above, just enter 'data'
- Output directory: When the training process ends, the best model weights and metrics log file will be saved in this directory. If a non-existing directory path is passed, it will be created.
- Epochs: Number of training epochs. (default=50)
- Batch size: Number of batch size. (default=2)
- Number of classes: How many classes we wish to segment. Can be 2/3/4/5. (default=3)
- Using Unet: Are we using Unet architecture or not? Passed as 0 for False and 1 for True (default=False i.e using ResNet)
- Train all – Are we training the entire network or just the last layers? (default=True)

If we wish to run an experiment, we must pass a data directory in the specific structure describe above and an output directory. The rest of the arguments are optional and will take default values unless passed explicitly.

Example (using the command line from the project main directory):
```
python main.py data output --epochs 50 --batchsize 2 --num_classes 5 --using_unet 0 --train_all 1
```

After running this command the training process initiates, which during you will see loss and different metrics statistics.
When the training ends, 2 files named 'weights.pt' and 'log.csv' will appear in your output directory - move 'weights.pt' to your 'weights' directory.


**Visualizing results:**

In order to visualize segmentation on training & test images, edit 'predict_all_data.py' file and do the following:
- In the last code line change the arguments of 'create_mask' to fit the arguments of the recently trained model.
- Run 'predict_all_data.py'. A pickle file named 'seg_results.p' will be created.
- Run 'helper.py' and wait for it to finish.

Now in your 'results' directory you can see all the train & test images; for each you can see the original image and mask, and the trained model segmentation of the image.

For a visualization of an end-to-end process please refer to the Visualization notebook.
