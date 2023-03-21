# Deformable Encoder Transformer (DEnT) 

## Train the pre-training model
```shell
cd dent/

# to run with default arguments
python pretrain.py

# to modify arguments
python pretrain.py --root <str> --dataroot <str> --world_size <int> --resume <bool/str> --train_folder <str> --val_folder <str> --epochs <int> --batch_size <int>
```

## Train the detection model
```shell
cd dent/

# to run with default arguments
python train.py

# to modify arguments
python train.py --root <str> --dataroot <str> --world_size <int> --resume <bool/str> --pretrain <bool> --pretrain_weights <str> --epochs <int> --nc <int> --r <int> --space <int> --train_batch <int> --val_batch <int>
```

## Run validation (only) on detection model
```shell
cd dent/

# to run with default arguments
python validate.py

# to modify arguments
python validate.py --root <str> --dataroot <str> --world_size <int> --weights <str> --nc <int> --r <int> --space <int> --batch <int>
```
