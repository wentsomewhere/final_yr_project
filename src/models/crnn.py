import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=37, nh=256):
        super(CRNN, self).__init__()
        self.imgH = imgH
        self.nc = nc
        self.nclass = nclass
        self.nh = nh
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        
        # RNN layers
        self.rnn = nn.LSTM(512, nh, bidirectional=True)
        
        # Output layer
        self.linear = nn.Linear(nh * 2, nclass)
        
    def forward(self, x):
        # Input validation
        if x.size(2) != self.imgH:
            x = F.interpolate(x, size=(self.imgH, x.size(3)), mode='bilinear', align_corners=False)
        
        # CNN
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        # RNN
        output, _ = self.rnn(conv)
        
        # Output layer
        output = self.linear(output)
        
        # Reshape to match expected dimensions
        output = output.permute(1, 0, 2)  # [b, w, nclass]
        output = output.reshape(-1, self.nclass)  # [b*w, nclass]
        
        return output
    
    def extract_features(self, x):
        """Extract features from the CNN layers."""
        features = []
        for layer in self.cnn:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features 