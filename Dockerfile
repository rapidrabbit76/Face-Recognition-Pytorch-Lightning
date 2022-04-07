FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /Training

RUN pip install --no-cache-dir \
    pytorch-lightning==1.5.7 \
    pytorch-metric-learning==1.1.1 \
    torchsummary==1.5.1 \ 
    torchmetrics==0.6.2 \ 
    opencv-python-headless==4.5.5.62 \ 
    albumentations==1.1.0 \ 
    mlflow==1.23.1 \ 
    pyarrow==7.0.0 \ 
    tensorboard==2.7.0 \ 
    boto3==1.21.4 \
    wandb==0.12.10 \
    python-dotenv 