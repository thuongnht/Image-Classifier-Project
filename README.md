# Image Classifier Project


## Data
Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). 
The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

You can download it using the following commands.

```python
!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
```


## Notebook Development
```
# run notebook
jupyter notebook

# run all cells in the notebook 'Image Classifier Project.ipynb'

```


## Command Line Application

### train.py
Train a new network on a dataset with train.py.
Prints out training loss, validation loss, and validation accuracy as the network trains

```
Basic usage:
    python train.py data_directory
Options:
    * Set directory to save checkpoints:
        python train.py data_dir --save_dir save_directory
    * Choose architecture:
        python train.py data_dir --arch "vgg13"
    * Set hyperparameters:
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * Use GPU for training:
        python train.py data_dir --gpu
   
Example:     
python train.py ./flowers --arch resnet152 --num_epochs 1 --save_dir models

```

### predict.py
Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

```
Basic usage:
    python predict.py path_image path_model
Options:
    * Return top KK most likely classes:
        python predict.py path_image path_model --topk 3
    * Use a mapping of categories to real names:
        python predict.py path_image path_model --category_names cat_to_name.json
    * Use GPU for inference:
        python predict.py path_image path_model --gpu
        
Example:
python .\predict.py flowers/test/1/image_06743.jpg ./models/image_classifier_vgg16.pth

```