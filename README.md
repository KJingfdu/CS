# CS
contrast learning based semi-supervised segmentation

### Usage

Build

```bash
conda create -n ntsd python=3.8 -y
conda activate ntsd
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install cityscapesScripts tensorboardX pyyaml opencv-python scipy scikit-image 
```

Train

```bash
# ours
python train_semi.py --config experiments/cityscapes/744/ours/config.yaml

# U2PL
python train_semi.py --config experiments/cityscapes/372/ours/config.yaml  # change contrastive method in the config file

# suponly
python train_semi.py --config experiments/cityscapes/372_pyr/suponly/config.yaml
```

Eval
```bash
python eval.py --model_path ckpt.pth
```

