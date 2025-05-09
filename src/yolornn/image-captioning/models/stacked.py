import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from models.modules.encoder_cnn import EncoderCNN
from models.modules.transformer import DecoderTransformer
from ultralytics import YOLO

class YOLO2Embed(nn.Module):
    def __init__(self, targ_embed_dim=512, in_channels=256):
        super(YOLO2Embed, self).__init__()
        self.downsampler = nn.Sequential(
            # First conv block - reduce channels
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Second conv block - further reduction
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global average pooling to handle variable spatial dimensions
            nn.AdaptiveAvgPool2d(1),
            
            # Final projection to target embedding dimension
            nn.Conv2d(64, targ_embed_dim, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, fmaps):
        return self.downsampler(fmaps)

class ViTCNNYOLO(nn.Module):
    def __init__(self):
        super(ViTCNNYOLO, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.head = nn.Identity() # Remove the classification head
        self.vit_transform = transforms.Compose([transforms.Resize((224, 224))])

        # todo: freeze ViT pretrained weights
        for param in self.vit.parameters():
            param.requires_grad = False

        # allow training of non pretrained CNN
        self.cnn = EncoderCNN()

        # freeze pretrained yolo
        yoloBig = YOLO("yolov8n.pt")
        self.yolo = yoloBig.model.model[0:9]

        for param in self.yolo.parameters():
            param.requires_grad = False

        self.yolo_downsampler = YOLO2Embed()

    def forward(self, images):
        vit_features = self.vit(self.vit_transform(images))
        cnn_features = self.cnn(images)

        # todo: check dims (or do we need to train a downscaling layer?)
        # yolo_feats = self.yolo(images)
        
        # (32,25600) below
        # yolo_feats = yolo_feats.view(yolo_feats.size(0), -1)
        
        #print(yolo_feats.shape
        concat_features = torch.cat((vit_features, cnn_features), dim=1)

        return concat_features


class VITCNNYOLOAttentionModel(DecoderTransformer):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_seq_length=50):
        super(VITCNNYOLOAttentionModel, self).__init__(embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length)
        self.vit = ViTCNNYOLO()
        self.vit_out_size = 1000 
        self.cnn_out_size = 2048

        self.fc_vit = torch.nn.Linear(1000, embed_size)
        self.fc_cnn = torch.nn.Linear(2048, embed_size)
        # self.fc_yolo = torch.nn.Linear(25600, embed_size)

        self.fc_scale = torch.nn.Linear(3*embed_size, 2*embed_size)

        self.yolo_downsampler = YOLO2Embed(targ_embed_dim=embed_size)

        self.VIT_SPLIT = (1/3)
        self.CNN_SPLIT = (1/3)
        self.YOLO_SPLIT = (1/3)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(2 * embed_size)
    
    def precompute_image(self, images):
        with torch.no_grad():
            enc_output = self.vit.forward(images)
        return enc_output
    
    def forward(self, images, captions, mode):
        with torch.no_grad():
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images) # shape: (batch_size, 8*64*64 + 2048)
        
        #print(enc_output.shape)
        vit_output = self.fc_vit(enc_output[:, :self.vit_out_size])
        cnn_output = self.fc_cnn(enc_output[:, self.vit_out_size:(self.vit_out_size+self.cnn_out_size)])
        # yolo_out = self.fc_yolo(enc_output[:, (self.cnn_out_size+self.vit_out_size):])
        yolo_feats = self.vit.yolo(images)
        # print(yolo_feats.shape)
        yolo_out = self.yolo_downsampler(yolo_feats)

        enc_output = torch.cat((vit_output, cnn_output, yolo_out), dim=1)
        enc_output = self.fc_scale(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.batchnorm(enc_output)
        enc_output = enc_output.unsqueeze(1)
        enc_output = enc_output.reshape(enc_output.size(0), 2, -1)
        #print(enc_output.shape)
        #print(captions.shape)
        return self.decoder_forward(enc_output, captions)
        
    def caption_images(self, images, vocabulary, mode="precomputed", max_length=40):
        self.eval()
        with torch.no_grad():
            # Encode the image
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images)
                
            #print(enc_output.shape)
            vit_output = self.fc_vit(enc_output[:, :self.vit_out_size])
            cnn_output = self.fc_cnn(enc_output[:, self.vit_out_size:(self.vit_out_size+self.cnn_out_size)])
            # yolo_out = self.fc_yolo(enc_output[:, (self.cnn_out_size+self.vit_out_size):])
            yolo_feats = self.vit.yolo(images)
            # print(yolo_feats.shape)
            yolo_out = self.yolo_downsampler(yolo_feats)

            enc_output = torch.cat((vit_output, cnn_output, yolo_out), dim=1)
            enc_output = self.fc_scale(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 2, -1)

            return self.greedy_inference(enc_output, vocabulary, max_length)

    def caption_images_beam_search(self, images, vocabulary, beam_width=3, mode="precomputed", max_length=50):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        # print("Batch size: ", batch_size)
        
        with torch.no_grad():
            # Encode all images in the batch
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images)

            #print(enc_output.shape)
            vit_output = self.fc_vit(enc_output[:, :self.vit_out_size])
            cnn_output = self.fc_cnn(enc_output[:, self.vit_out_size:(self.vit_out_size+self.cnn_out_size)])
            # yolo_out = self.fc_yolo(enc_output[:, (self.cnn_out_size+self.vit_out_size):])
            yolo_feats = self.vit.yolo(images)
            # print(yolo_feats.shape)
            yolo_out = self.yolo_downsampler(yolo_feats)

            enc_output = torch.cat((vit_output, cnn_output, yolo_out), dim=1)
            enc_output = self.fc_scale(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 2, -1)
            enc_output = enc_output.unsqueeze(1).expand(-1, beam_width, -1, -1)
            enc_output = enc_output.reshape(batch_size * beam_width, 2, -1)
            
            return self.beam_search_inference(enc_output, batch_size, vocabulary, beam_width, max_length)
