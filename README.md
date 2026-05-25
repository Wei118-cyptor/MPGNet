# MPGNet
# MPGNet: Multi-modal Prototype-Guided Discriminative Feature Learning for Transmission Line Fault Detection

<!-- <p align="center"> <img src='docs/teaser.jpg' align="center"> </p> -->

## Installation
```shell script
conda create --name MPGNet python=3.8 -y && conda activate MPGNet
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/Wei118-cyptor/MPGNet.git
```

Install detectron2 and other dependencies
```shell script
cd MPGNet/third_party/detectron2
pip install -e .
cd ../..
pip install -r requirements.txt
```
