import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# spatial transformer 

class Tnet(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.conv0 = nn.Conv1d(n, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n * n)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.n = n

    def forward(self, input):
        out = F.relu(self.bn0(self.conv0(input)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = torch.max(input=out, dim=2, keepdim=True)[0]
        out = out.view(-1, 1024)
        out = F.relu(self.bn3(self.fc0(out)))
        out = F.relu(self.bn4(self.fc1(out)))
        out = self.fc2(out)
        weird_ass_thing = torch.from_numpy(np.eye(self.n).flatten().astype(np.float32)).view(-1, self.n * self.n).repeat(input.size()[0], 1)
        out = out + weird_ass_thing
        out = out.view(-1, self.n, self.n)
        return out

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transformation = Tnet()
        self.feature_transformation = Tnet(n=64)
        self.conv0 = nn.Conv1d(3, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, input):
        x1 = self.input_transformation(input)
        out = torch.bmm(torch.transpose(input, 1, 2), x1).transpose(1, 2)
        out = F.relu(self.bn0(self.conv0(out)))
        x2 = self.feature_transformation(out)
        out = torch.bmm(torch.transpose(out, 1, 2), x2).transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)
        out=out.type(torch.FloatTensor)
        x1=x1.type(torch.FloatTensor)
        x2=x2.type(torch.FloatTensor)
        return out, x1, x2

class EdgeConv(nn.Module):
  def __init__(self, k, in_features, out_features):
    super().__init__()
    self.k = k
    self.in_features = in_features
    self.out_features = out_features
  
  def knn(self, input):
    inner = -2 * torch.matmul(input.transpose(2,1).contiguous(), input)
    xx = torch.sum(input ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2,1).contiguous()
    idx = pairwise_distance.topk(self.k, dim=-1)[1]  # (batch_size, num_points, k)
    idx_base = torch.arange(0, 103).view(-1, 1, 1)*31
    idx = idx + idx_base
    batch_size=103
    num_points=31
    idx = idx.view(-1)
    _, num_dims, _ = input.size()
    x=input
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, self.k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature
  
  def forward(self, input):
    input = self.knn(input)
    self.edge_conv = nn.Sequential(
        nn.Conv2d(self.in_features, self.out_features,1),
        nn.LeakyReLU(0.1),
        nn.BatchNorm2d(self.out_features), # not sure about this number
    )
    out = self.edge_conv(input)
    out=out.max(dim=-1,keepdim=False)[0]
    return out

class DGCNN_encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.edgeconv1 = EdgeConv(20, 6, 64)
    self.edgeconv2 = EdgeConv(20, 128, 64)
    self.edgeconv3 = EdgeConv(20, 128, 128)
    self.edgeconv4 = EdgeConv(20, 256, 256)
    self.transform = Transform()
  
  def forward(self,input):
    out,x1,x2 = self.transform(input)
    new_out=torch.matmul(x1,input)
    out1 = self.edgeconv1(new_out)
    out2 = self.edgeconv2(out1)
    out3 = self.edgeconv3(out2)
    out4 = self.edgeconv4(out3)
    encoded = torch.cat((out1, out2, out3, out4), 1)
    return encoded

input =torch.rand(103,3,31)
Final_Encoder=DGCNN_encoder()
output_finale=Final_Encoder(input)
print(output_finale.shape)

