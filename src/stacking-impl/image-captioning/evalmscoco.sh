# embed size can be 512/1024/2048
# lets just eval all to be sure
python eval.py --batch_size=128 --learning_rate=0.001 --embed_size=512 \
               --num_layers=4 --model_arch=stacked --dataset=mscoco \
               --checkpoint_dir=/storage/ice1/0/7/agupta965/checkpoints/stacked/flickr/bs128_lr0.001_es512_nl4/checkpoint_epoch_75.pth.tar