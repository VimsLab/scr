# Deformable Encoder Transformer (DEnT) 

## Train the pre-training model
```
cd dent/
# to run with default arguments
python pretrain.py
```
Arguments
```
root &#9; <path/to/root (str)> 
dataroot &#9; <path/to/dataset (str)> 
world_size &#9; <world size (int)>
resume &#9; <False (bool) or path/to/trained_weights (str)>
train_folder &#9; <dir_name with train dataset (str)> 
val_folder &#9; <dir_name with val dataset (str)>
epochs &#9; <num epochs (int)>
batch_size &#9; <batch size (int)>  


## Train the detection model
```
cd dent/
# to run with default arguments
python train.py
```
Arguments
```
root &#9; <path/to/root (str)> 
dataroot &#9; <path/to/dataset (str)> 
world_size &#9; <world size (int)>
resume &#9; <False (bool) or path/to/trained_weights (str)>
pretrain &#9; <use pretrained weights? (bool)>
pretrain_weights &#9; <path/to/pre_trained/weights (str)>
epochs &#9; <num epochs (int)>
nc &#9; <num of classes (int)>
r &#9; <num of adjacent images to stack (int)>
space &#9; <num of steps/ stride for next adjacent image block (int)>
train_batch &#9; <training batch size (int)>  
val_batch &#9; <validation batch size (int)>


## Run validation (only) on detection model
```
cd dent/
# to run with default arguments
python validate.py
```
Arguments
```
root &#9; <path/to/root (str)> 
dataroot &#9; <path/to/dataset (str)> 
world_size &#9; <world size (int)>
weights &#9; <path/to/trained/weights (str)>
nc &#9; <num of classes (int)>
r &#9; <num of adjacent images to stack (int)>
space &#9; <num of steps/ stride for next adjacent image block (int)>
batch &#9; <batch size (int)>  

