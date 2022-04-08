# Face Recognition Pytorch Lightning

Implementation of Face Recognition models using pytorch lightning

# DATASET

- train: CASIA-WebFace (10575 class)
- validation/test: LFW

# Requirements

need for training

- pytorch-lightning==1.5.7
- pytorch-metric-learning==1.1.1
- torchsummary==1.5.1
- torchmetrics==0.6.2
- opencv-python-headless==4.5.5.62
- albumentations==1.1.0
- mlflow==1.23.1
- pyarrow==7.0.0
- tensorboard==2.7.0
- boto3==1.21.4
- wandb==0.12.10

need for test

- onnxruntime==1.10.0
- pytest==7.0.0
- pytest-watch==4.2.0
- pytest-testmon==1.2.3
- pytest-cov==3.0.0

# Training

```bash
# Training MobileFaceNet, CASIA dataset using base transforms

python main.py \
--dataset="casia" \
--num_classes=10575 \
--transforms="base" \
--backbone="MobileFaceNet" \
--train_data_dir="{CASIA_DATASET_PATH}" \
--validation_data_dir="{LFW_DATASET_PATH}" \
--log_every_n_steps=5 \
--check_val_every_n_epoch=1 \
--batch_size=128 \
--max_epochs=600 \
--optimizer="SGD" \
--lr=0.1 \
--weight_decay=0.0005 \
--momentum=0.9 \
--lr_step_gamma=0.1 \
--m=0.5 \
--s=32.0 \
--gpus=1 \
--precision=16 \
--callbacks_monitor="val/acc" \
--checkpoint_top_k=3 \
--callbacks_refresh_rate=1 \
--detect_anomaly=True

```

# Supported backbone

- MobileFaceNet
- Convolution Block Attention Module
- AttentionResnet

# Result

| Backbone      | ACC(LFW) |
| ------------- | -------- |
| MobileFaceNet | 95.4     |

# Pretrained model

| Backbone      | torchscript                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| MobileFaceNet | [link](https://drive.google.com/file/d/1ksVfEBQq8UHadS2tN_WnUzu8m7HGfxxC/view?usp=sharing) |

# References

- https://github.com/TreB1eN/InsightFace_Pytorch
