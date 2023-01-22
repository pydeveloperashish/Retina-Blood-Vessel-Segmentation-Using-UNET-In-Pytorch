import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(num_features = out_channel)
        
        self.conv2 = nn.Conv2d(in_channels = out_channel, out_channels = out_channel, 
                               kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(num_features = out_channel)
        
          
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x



class encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = conv_block(in_channel = in_channel, out_channel = out_channel)
        self.pool = nn.MaxPool2d(kernel_size = (2, 2))
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        
        return x, p
    
    
class decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels = in_channel,
                                     out_channels = out_channel,
                                     kernel_size = 2,
                                     stride = 2,
                                     padding = 0)
        
        self.conv = conv_block(in_channel = out_channel + out_channel,
                               out_channel = out_channel)
        
    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], axis = 1)
        x = self.conv(x)
        return x




class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """ Encoder """
        self.e1 = encoder_block(in_channel = 3, out_channel = 64)
        self.e2 = encoder_block(in_channel = 64, out_channel = 128)
        self.e3 = encoder_block(in_channel = 128, out_channel = 256)
        self.e4 = encoder_block(in_channel = 256, out_channel = 512)


        """ BottleNeck """
        self.b = conv_block(in_channel = 512, out_channel = 1024)
        
        """ Decoder """
        self.d1 = decoder_block(in_channel = 1024, out_channel = 512)
        self.d2 = decoder_block(in_channel = 512, out_channel = 256)
        self.d3 = decoder_block(in_channel = 256, out_channel = 128)
        self.d4 = decoder_block(in_channel = 128, out_channel = 64)
        
        
        """ Classifier """
        self.outputs = nn.Conv2d(in_channels = 64, out_channels = 1,
                                 kernel_size = 1, padding = 0)
        

    def forward(self, x):
        """ Encoder"""
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        """ BottleNeck """
        b = self.b(p4)
        #print(f"Input Shape {x.shape}")
        #print(s1.shape, s2.shape, s3.shape, s4.shape)
        #print(b.shape)
        
        """ Decoder """
        d1 = self.d1(b, skip_connection = s4)
        d2 = self.d2(d1, skip_connection = s3)
        d3 = self.d3(d2, skip_connection = s2)
        d4 = self.d4(d3, skip_connection = s1)
        
        
        """ Classifier """
        outputs = self.outputs(d4)
        
        return outputs

      


if __name__ == '__main__':
    x = torch.randn((2, 3, 512, 512))   # BS, channels, img_size, img_size
    f = build_unet()
    y = f(x)
    #print(y.shape) 