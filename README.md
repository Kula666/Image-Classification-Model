# Image-Classification-Model
A common model of CNN image classification is implemented, and the framework can be reused to a new network

## 1. Requirements
* python (>=3.6)
* pytorch (>=1.1.0)
* torchvision (>= 0.3.0)
* other dependencies run
```bash
pip install -r requirements.txt
```

<br>

## 2. Train
### （1）Use GPU
For example, we will use GPU to train alexnet.
Modify `experiments/cifar-10/alexnet/config.yaml` under device to cuda.
```yaml
device: cuda
```
Then run the following command to start training from scratch
```bash
python train.py -p ./experiments/cifar-10/alexnet
```
If you want to continue the last training
```bash
python train.py -p ./experiments/cifar-10/alexnet -r
```
### （2）Use CPU
For example, we will use CPU to train alexnet.
Modify `experiments/cifar-10/alexnet/config.yaml` under device to cpu.
```yaml
device: cpu
```
Then run the following command to start training from scratch
```bash
python train.py -p ./experiments/cifar-10/alexnet
```
If you want to continue the last training
```bash
python train.py -p ./experiments/cifar-10/alexnet -r
```

<br>


## 3. Train own network
（1）Add your own network structure to the model directory and write a function that returns this network object, passing num_class
```python
def vgg19(num_classes):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)
```

（2）Add your own network in `model/__init__.py`
```
from vgg import *
```

（3）Copy the configuration file and adjust the parameters

（4）Then run the following to train
```bash
python train.py -p The location of the configuration file
```
