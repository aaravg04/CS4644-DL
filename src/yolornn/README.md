# YOLO-RNN Model Implementation
Model is YOLO -> RNN (LSTM), code in `image-captioning/models/yolornn_model.py`. 

Reference `download-datasets.sh` for downloading the flickr30k/mscoco datasets, `image-captioning/configs/yolornn-mscoco.yaml` for configs used during training/testing process

See `sample_train_command*.sh` and `wandbtrain.sh` for commands used for test training runs, `train.py` for the provided train implementation and `train_wandb.py` for the modified version used for logging metrics on WeightsAndBiases during training.

Scripts like `req_interactive.sh`, `evaljob.sh`, `newmain.sh` were to submit SLURM jobs on GT's PACE-ICU cluster to allocate resources for our training jobs.

For training we submitted train jobs via slurm jobs with the config in `image-captioning/configs/stack-*.yaml` and logged eval metrics like BLEU, METEOR, CIDER scores to WandB during the training process for easy evaluation. 

Our model implementation for the YOLO-RNN model is in yolornn_model.py and it currently reflects the version with LSTM to implement a baseline model and get metrics that we later improve with our stacked encoder architecture.
