##This is a pytorch implement for DB-FusionNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import spectral
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,recall_score,cohen_kappa_score,accuracy_score
from scipy.io import loadmat
import os
from tqdm.notebook import tqdm
from PIL import Image
import math
import os
from scipy.io import loadmat
from PIL import Image
from spectral import open_image, envi  # Ensure envi is correctly imported

##hypeperameters and experimental settings
RANDOM_SEED=42
DATASET = 'SA'    ## PU  IP  SA  
TRAIN_RATE = 0.05  ## ratio of training data
VAL_RATE = 0.1    ## ratio of valuating data
EPOCH = 100   ##number of epoch
VAL_EPOCH = 1  ##interval of valuation
LR = 0.0005    ##learning rate
WEIGHT_DECAY = 1e-6  
BATCH_SIZE = 256
DEVICE = 0  ##-1:CPU  0:cuda 0
N_PCA = 15  ## reserved PCA components
PATCH_SIZE =25 
SAVE_PATH = f"results\\{DATASET}"
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)


## Set random seed for reproduction
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def loadData(name):
    data_path = os.path.join(os.getcwd(), 'dataset')
    if name == 'IP':
        data = loadmat(os.path.join(data_path, 'IndianPines', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, 'IndianPines', 'Indian_pines_gt.mat'))['indian_pines_gt']
        class_name = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", 
                      "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
    elif name == 'SA':
        data = loadmat(os.path.join(data_path, 'Salinas', 'Salinas_corrected.mat'))['salinas_corrected']
        labels = loadmat(os.path.join(data_path, 'Salinas', 'Salinas_gt.mat'))['salinas_gt']
        class_name = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical']
    elif name == 'PU':
        data = loadmat(os.path.join(data_path, 'PaviaU', 'PaviaU.mat'))['paviaU']
        labels = loadmat(os.path.join(data_path, 'PaviaU', 'PaviaU_gt.mat'))['paviaU_gt']
        class_name = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 
                      'Bitumen', 'Self-Blocking Bricks', 'Shadows']
    elif name == 'longkou':
        data = loadmat(os.path.join(data_path, 'Longkou', 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = loadmat(os.path.join(data_path, 'Longkou', 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']  # Note: key name may need to be adjusted according to actual situation
        class_name = ['corn', 'cotton', 'sesame', 'broad-leaf soybean', 'narrow-leaf soybean', 'rice','Water','Roads and houses','Mixed weed']
    
    elif name == 'hanchuan':
        data = loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']  # Note: key name may need to be adjusted according to actual situation
        class_name = ['strawberry', 'cowpea', 'soybean', 'sorghum', 'water spinach', 'watermelon','greens','trees','grass','red roof','gray roof','plastic','bare soil','road','bright object','water']
    
    elif name == 'honghu':
        data = loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']  # Note: key name may need to be adjusted according to actual situation
        class_name = ['red roof', 'road', 'bare soil', 'cotton', 'cotton firewood', 'rape','chinese cabbage','pakchoi','cabbage','tuber mustard','brassica parachinensis','brassica chinensis','small brassica chinensis','lactuca sativa','celtuce','film covered letture','romaine letture','carrot','white radish','garlic sprout','broad bean','tree']
    
    elif name == 'plot':
        # Read ENVI format hyperspectral data
        hdr_file = os.path.join(data_path, 'plot1000', 'cropped.hdr')
        img = spectral.open_image(hdr_file)
        
        # Convert data to numpy array
        data = np.array(img.load())
        
        # Read PNG label image
        png_label_path = os.path.join(data_path, 'plot1000', 'Label_1_rgb_image.png')
        label_image = Image.open(png_label_path)
        labels = np.array(label_image)
        class_name = ['soybean','soil']
        
    elif name == 'nonggaoqu':
        # Read ENVI format hyperspectral data
        hdr_file = os.path.join(data_path, 'nonggaoqu', 'sg.hdr')
        img = spectral.open_image(hdr_file)
        
        # Convert data to numpy array
        data = np.array(img.load())
        
        # Read PNG label image
        png_label_path = os.path.join(data_path, 'nonggaoqu', 'label.png')
        label_image = Image.open(png_label_path)
        labels = np.array(label_image)
        class_name = ['class_1','class_2','class_3','class_4','class_5']
    elif name == 'XiongAn':
        from spectral import envi  # Ensure envi module is imported
        # Define dataset
        data_path = os.path.join(os.getcwd(), 'dataset', 'XiongAn')
        hdr_file = os.path.join(data_path, 'XiongAn.hdr')
        img_file = os.path.join(data_path, 'XiongAn.img')  # Read corresponding .img file

       
        # Define labels
        label_path = os.path.join(os.getcwd(), 'dataset', 'XiongAn')
        label_hdr_file = os.path.join(label_path, 'farm_roi.hdr')
        label_img_file = os.path.join(label_path, 'farm_roi.img')  # Read corresponding .img file

        label_img = envi.open(label_hdr_file)  # Open hdr file
        labels = np.array(label_img.load())  # Load data as numpy array
        # Open data file and load data
        data_img = envi.open(hdr_file)  # Open data HDR file
        data = np.array(data_img.load())  # Load data as numpy array

  
       
        class_name = ['Rice stubble', 'Grassland', 'Elm', 'White ash', 'Chinese honeylocust', 'Vegetable field', 'Poplar', 'Soybean', 'Acacia', 'Rice',' Water body', 'Willow, Maple',' Golden rain tree', 'Peach tree',' Corn', 'Pear tree', 'Lotus leaf', 'Building']
    
    return data, labels, class_name


data,label,class_name = loadData(DATASET)
NUM_CLASS = label.max()

# Check shape
print(f"data shape: {data.shape}")
print(f"label shape: {label.shape}")
# Adjust label shape from (1580, 3750, 1) to (1580, 3750)
labels = np.squeeze(label)
# Check shape again
print(f"Adjusted label shape: {labels.shape}")

# Ensure shape matching
if data.shape[:2] != labels.shape:
    raise ValueError("The shapes of data and label do not match!")


# Display HSI
rgb_view = spectral.imshow(data, (130,65,20), classes=label, title='RGB origin', figsize=(7, 7))
gt_view = spectral.imshow(classes=label, title='GroundTruth', figsize=(7, 7))
view = spectral.imshow(data, (130, 65, 20), classes=label, figsize=(7, 7))
view.set_display_mode('overlay')
view.class_alpha = 0.5
view.set_title('Overlay')



# Save images
DATASET = 'your_dataset_name'
# Save original data RGB image
spectral.save_rgb(os.path.join(SAVE_PATH, f"{DATASET}_RGB_origin.jpg"), data, (130, 65, 20))

# Save label RGB image
spectral.save_rgb(os.path.join(SAVE_PATH, f"{DATASET}_RGB_gt.jpg"), label, colors=spectral.spy_colors)

def applyPCA(X, numComponents=15):
    """PCA processing

    Args:
        X (ndarray M*N*C): data needs DR
        numComponents (int, optional): number of reserved components. Defaults to 15.

    Returns:
        newX: _description_
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)   ##PCA and normalization
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

data,pca = applyPCA(data,N_PCA)
data.shape

def sample_gt(gt, train_rate):
    """ generate training gt for training dataset
    Args:
        gt (ndarray): full classmap
        train_rate (float): ratio of training dataset
    Returns:
        train_gt(ndarray): classmap of training data
        test_gt(ndarray): classmap of test data
    """
    indices = np.nonzero(gt)  ##([x1,x2,...],[y1,y2,...])
    X = list(zip(*indices))  ## X=[(x1,y1),(x2,y2),...] location of pixels
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_rate > 1:
       train_rate = int(train_rate)
    train_indices, test_indices = train_test_split(X, train_size=train_rate, stratify=y, random_state=100)
    train_indices = [t for t in zip(*train_indices)]   ##[[x1,x2,...],[y1,y2,...]]
    test_indices = [t for t in zip(*test_indices)]
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    
    return train_gt, test_gt

train_gt, test_gt = sample_gt(label,TRAIN_RATE)
val_gt,test_gt = sample_gt(test_gt,VAL_RATE/(1-TRAIN_RATE))

## display sampling info
sample_report = f"{'class': ^10}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(label):
    if i == 0: continue
    sample_report += f"{i: ^10}{(train_gt==i).sum(): ^10}{(val_gt==i).sum(): ^10}{(test_gt==i).sum(): ^10}{(label==i).sum(): ^10}\n"
sample_report += f"{'total': ^10}{np.count_nonzero(train_gt): ^10}{np.count_nonzero(val_gt): ^10}{np.count_nonzero(test_gt): ^10}{np.count_nonzero(label): ^10}"
print(sample_report)
spectral.imshow(classes=train_gt, title='train_gt')
spectral.imshow(classes=val_gt, title='val_gt')
spectral.imshow(classes=test_gt, title='test_gt')

class PatchSet(Dataset):
    """ Generate 3D patch from hyperspectral dataset """
    def __init__(self, data, gt, patch_size, is_pred=False):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the 3D patch
            is_pred: bool, create data without label for prediction (default False) 

        """
        super(PatchSet, self).__init__()
        self.is_pred = is_pred
        self.patch_size = patch_size
        p = self.patch_size // 2
        self.data = np.pad(data,((p,p),(p,p),(0,0)),'constant',constant_values = 0)
        if is_pred:
            gt = np.ones_like(gt)
        self.label = np.pad(gt,(p,p),'constant',constant_values = 0)
        x_pos, y_pos = np.nonzero(gt)
        x_pos, y_pos = x_pos + p, y_pos + p   ##indices after padding
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]
        data = np.asarray(data, dtype='float32').transpose((2, 0, 1))
        label = np.asarray(label, dtype='int64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        if self.is_pred:
            return data
        else: return data, label

##create dataset and dataloader
train_data = PatchSet(data, train_gt, PATCH_SIZE)
val_data = PatchSet(data, val_gt, PATCH_SIZE)
all_data = PatchSet(data, label, PATCH_SIZE,is_pred = True)
train_loader = DataLoader(train_data,BATCH_SIZE,shuffle= True)
val_loader = DataLoader(val_data,BATCH_SIZE,shuffle= True)
all_loader = DataLoader(all_data,BATCH_SIZE,shuffle= False)

d,g=train_data.__getitem__(0)
d.shape,g


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to add attention mechanism."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

# ECA Block for 1D CNN
class ECABlock1D(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        """
        ECA Block for 1D Convolutional layers.
        Args:
            channels: Number of input channels (depth of the feature map).
            b: Parameter for kernel size computation.
            gamma: Parameter for kernel size computation.
        """
        super(ECABlock1D, self).__init__()
        
        # Compute kernel size using the formula
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Global Average Pooling 1D
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 1D Convolution with the computed kernel size
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)

    def forward(self, x):
        # Get input shape
        b, c, h = x.size()
        
        # Global average pooling [B, C, H] -> [B, C, 1]
        y = self.avg_pool(x)
        
        # Reshape to [B, 1, C] for Conv1d processing
        y = y.view(b, 1, c)
        
        # Apply Conv1d [B, 1, C] -> [B, 1, C]
        y = self.conv1(y)
        
        # Apply sigmoid activation
        y = torch.sigmoid(y)
        
        # Reshape back to [B, C, 1]
        y = y.view(b, c, 1)
        
        # Multiply with the input feature map
        return x * y.expand_as(x)


class HybridSN(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums):
        super().__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        
        # 3D convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(7, 3, 3)),  # Reduce channels
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5, 3, 3)),  # Reduce channels
            nn.ReLU(inplace=True)
        )
        
        # Calculate 3D convolution output shape
        self.x1_shape = self.get_shape_after_3dconv()

        # 2D convolutional layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.x1_shape[1] * self.x1_shape[2], out_channels=32, kernel_size=(3, 3)),  # Reduce channels
            nn.ReLU(inplace=True)
        )
        
        # SPPF module
        self.sppf = SPPF(32, 64)  # Reduce output channels
        
        # 1D convolutional layers
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=15, out_channels=16, kernel_size=3),  # Reduce channels
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3),  # Reduce channels
            nn.ReLU(inplace=True)
        )

        # Calculate flattened size
        x1_flat_size = self.get_flatten_size_after_sppf()
        x2_flat_size = self.get_flatten_size_after_1dconv()
        
        # Fully connected layers
        self.dense1 = nn.Sequential(
            nn.Linear(x1_flat_size + x2_flat_size, 128),  # Reduce input dimension
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(128, 64),  # Reduce output dimension
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        self.dense3 = nn.Sequential(
            nn.Linear(64, class_nums)
        )
    
    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
        return x.shape
    
    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv3(x)
        return x.shape

    def get_flatten_size_after_sppf(self):
        """Calculate flattened size after SPPF"""
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv3(x)
            x = self.sppf(x)
        return x.view(1, -1).shape[1]

    def get_flatten_size_after_1dconv(self):
        """Calculate flattened size after 1D convolution"""
        x = torch.zeros((1, 15, self.patch_size * self.patch_size))
        with torch.no_grad():
            x = self.conv4(x)
            x = self.conv5(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, X):
        # First path: 3D convolution + 2D convolution + SPPF
        x1 = X.unsqueeze(1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = x1.view(x1.shape[0], x1.shape[1] * x1.shape[2], x1.shape[3], x1.shape[4])
        x1 = self.conv3(x1)
        x1 = self.sppf(x1)  # SPPF operation
        x1 = x1.view(x1.size(0), -1)  # Flatten

        # Second path: 1D convolution
        x2 = X.view(X.size(0), 15, -1)  # reshape for Conv1d
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x2 = x2.view(x2.size(0), -1)  # Flatten

        # Concatenate outputs from both paths
        x = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out


# Create model
net = HybridSN(N_PCA, PATCH_SIZE, class_nums=NUM_CLASS)

# View model summary
summary(net, input_size=(1, N_PCA, PATCH_SIZE, PATCH_SIZE), 
        col_names=['num_params', 'kernel_size', 'mult_adds', 'input_size', 'output_size'],
        col_width=10, row_settings=['var_names'], depth=4)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Assume data is an iterable object

    def __len__(self):
        return len(self.data)  # Return dataset size

    def __getitem__(self, index):
        sample = self.data[index]  # Get sample
        return self.process_tensor(sample)  # Process sample

    def process_tensor(self, tensor):
        target_size = 25
        current_size = tensor.shape[0]

        if current_size < target_size:
            # Padding
            padding = target_size - current_size
            return F.pad(tensor, (0, 0, 0, padding), 'constant', 0)
        elif current_size > target_size:
            # Cropping
            return tensor[:target_size]
        else:
            return tensor  # If already target size, return directly

# Ensure each sample's first dimension is 25
sample_data = [torch.randn(25, 25, 15) for _ in range(100)]  # Each sample size is [25, 25, 15]

# Create dataset and data loader
dataset = MyDataset(sample_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test data loader
for batch_idx, data in enumerate(data_loader):
    print(f"Batch {batch_idx} - Data shape: {data.shape}")


## training the model
device = torch.device(DEVICE if DEVICE>=0 and torch.cuda.is_available() else 'cpu')
loss_list = []
acc_list = []
val_acc_list = []
val_epoch_list = []

model = HybridSN(N_PCA,PATCH_SIZE,class_nums=NUM_CLASS)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),LR,weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_func = nn.CrossEntropyLoss()
batch_num = len(train_loader)
train_num = train_loader.dataset.__len__()
val_num = val_loader.dataset.__len__()

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
try:
    for e in tqdm(range(EPOCH), desc="Training:"):
        model.train()
        avg_loss = 0.
        train_acc = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=batch_num):
            data,target = data.to(device),target.to(device)
            optimizer.zero_grad()
            out = model(data)
            target = target - 1  ## class 0 in out is class 1 in target
            loss = loss_func(out,target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            _,pred = torch.max(out,dim=1)
            train_acc += (pred == target).sum().item()
        loss_list.append(avg_loss/train_num)
        acc_list.append(train_acc/train_num)
        print(f"epoch {e}/{EPOCH} loss:{loss_list[e]}  acc:{acc_list[e]}")
        ## valuation
        if (e+1)%VAL_EPOCH == 0 or (e+1)==EPOCH:
            val_acc =0
            model.eval()
            for batch_idx, (data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
                data,target = data.to(device),target.to(device)
                out = model(data)
                target = target - 1  ## class 0 in out is class 1 in target
                _,pred = torch.max(out,dim=1)
                val_acc += (pred == target).sum().item()
            val_acc_list.append(val_acc/val_num)
            val_epoch_list.append(e)
            print(f"epoch {e}/{EPOCH}  val_acc:{val_acc_list[-1]}")
            save_name = os.path.join(SAVE_PATH, f"epoch_{e}_acc_{val_acc_list[-1]:.4f}.pth")
            torch.save(model.state_dict(),save_name)
    ax1.plot(np.arange(e+1),loss_list)
    ax1.set_title('loss')
    ax1.set_xlabel('epoch')
    ax2.plot(np.arange(e+1),acc_list,label = 'train_acc')
    ax2.plot(val_epoch_list,val_acc_list,label = 'val_acc')
    ax2.set_title('acc')
    ax2.set_xlabel('epoch')
    ax2.legend()
except Exception as exc:
    print(exc)
finally: 
    print(f'Stop in epoch {e}')

