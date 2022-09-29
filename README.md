# Boosting Dense Long-Tailed Object Detection from Data-Centric View
Pytorch implementation of REDet, ACCV 2022
## Getting start

### 1. Download pretrained models.

```
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
```

After downloading, please put it into "pretrained/" 

### 2. Prepare  Dataset (LVIS v1)

Then download the images and annotations from the official website of [LVIS](https://www.lvisdataset.org/dataset). 

Finally the file structure of folder `lvis` will be like this:

```
$lvis
  ├── annotations
  │   ├── lvis_v1_val.json
  │   ├── lvis_v1_train.json
  ├── train2017
  │   ├── 000000004134.png
  │   ├── 000000031817.png
  │   ├── ......
  ├── val2017
  ├── test2017
```

### 3. Prepare the Environment

```shell
# create environment
conda create --name REDet python=3.7
conda activate REDet

# install pytorch
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1
```

edit `easy_setup.sh`：

```shell
#!/bin/bash

export PATH=/your/path/to/gcc-5.3.0/bin/:$PATH # gcc path
export LD_LIBRARY_PATH=/your/path/to/gmp-4.3.2/lib/:/your/path/to/mpfr-2.4.2/lib/:/your/path/to/mpc-0.8.1/lib/:$LD_LIBRARY_PATH # lib path
export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0' # cuda list

python setup.py build_ext -i
```

Then:

```shell
# setup
./easy_setup.sh

# pip install requirements.txt
pip install -r requirements.txt

# install other packages
pip uninstall protobuf
pip install protobuf==3.20.1
pip install pyyaml
pip install scikit-image
```

### 4. Run the code
