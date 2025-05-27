import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from PIL import Image

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = F.softmax(attention_weights, dim=1)
        return torch.sum(hidden_states * attention_weights, dim=1)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, language='en'):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        # Language-specific character sets
        self.language_chars = {
            'en': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~',
            'hi': 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕ६७८९',
            'zh': '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞',
        }
        
        self.language = language
        self.char_set = self.language_chars.get(language, self.language_chars['en'])
        nclass = len(self.char_set) + 1  # +1 for blank token

        # CNN layers
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )
        self.attention = AttentionModule(nh * 2)

    def forward(self, input):
        # CNN features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # RNN features
        output = self.rnn(conv)
        
        # Apply attention
        output = self.attention(output)
        
        return output

    def get_attention_map(self, input):
        """Generate attention heatmap for explainable AI."""
        # Get CNN features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        
        # Get attention weights
        conv_flat = conv.view(b, c, -1).permute(0, 2, 1)
        attention_weights = self.attention(conv_flat)
        
        # Reshape attention weights to match input size
        attention_map = attention_weights.view(b, 1, h, w)
        attention_map = F.interpolate(attention_map, size=input.shape[-2:], mode='bilinear', align_corners=False)
        
        return attention_map

class OCRModule:
    def __init__(self, device='cuda', language='en'):
        self.device = device
        self.language = language
        self.model = CRNN(
            imgH=32,
            nc=3,
            nclass=95,  # Will be updated based on language
            nh=256,
            language=language
        ).to(device)
        
        # Character set for decoding
        self.char_set = self.model.char_set
        
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        
    def predict(self, image, return_attention=False):
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            
            if return_attention:
                attention_map = self.model.get_attention_map(image)
                return output, attention_map
            return output
        
    def decode_predictions(self, output):
        """Decode model output to text using CTC decoding."""
        # Convert logits to probabilities
        probs = F.softmax(output, dim=-1)
        
        # Get most likely character for each time step
        _, preds = torch.max(probs, dim=-1)
        
        # Convert predictions to text
        texts = []
        for pred in preds:
            # Remove duplicates and blank tokens
            char_list = []
            prev_char = None
            for char_idx in pred:
                if char_idx != 0 and char_idx != prev_char:  # 0 is blank token
                    char_list.append(self.char_set[char_idx - 1])
                prev_char = char_idx
            
            # Join characters into text
            text = ''.join(char_list)
            texts.append(text)
        
        return texts[0] if len(texts) == 1 else texts
    
    def generate_attention_visualization(self, image, attention_map):
        """Generate visualization of attention heatmap overlaid on input image."""
        # Convert attention map to numpy
        attention_map = attention_map.cpu().numpy()[0, 0]
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        
        # Convert input image to numpy
        image_np = image.cpu().numpy()[0].transpose(1, 2, 0)
        image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        
        # Overlay heatmap on image
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
        
        return Image.fromarray(overlay) 