# Stacked Model Implementation
Model is | ViT | CNN | YOLOv8n -> | Dense / ConvBlock | | -> Dense -> Transformer Decoder, code in `image-captioning/models/stacking.py`. 

Reference `download-datasets.sh` for downloading the flickr30k/mscoco datasets, `image-captioning/configs/stack-*.yaml` for configs used during training/testing process

See `sample_train_command*.sh` and `wandbtrain.sh` for commands used for test training runs, `train.py` for the provided train implementation and `train_wandb.py` for the modified version used for logging metrics on WeightsAndBiases during training.

Scripts like `req_interactive.sh`, `evaljob.sh`, `newmain.sh` were to submit SLURM jobs on GT's PACE-ICU cluster to allocate resources for our training jobs.

For training we submitted train jobs via slurm jobs with the config in `image-captioning/configs/stack-*.yaml` and logged eval metrics like BLEU, METEOR, CIDER scores to WandB during the training process for easy evaluation. 

Our model implementation for the stacked model is in stacked.py and it currently reflects the version with the CNN downsampler; the Linear downsampler is in a previous version, in future work we hope to make an easier to use way to swap between stacked-linear-downsampler, stacked-cnn-downsampler models during training and evaluation.
