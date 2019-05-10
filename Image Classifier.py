#!/usr/bin/env python
# coding: utf-8

# <img src='assets/Flowers.png' width=500px>


# In[1]:



get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.models import vgg16
from PIL import Image
import numpy as np
import json
from collections import OrderedDict





train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

vt_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder('./flowers/'+train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder('./flowers/'+valid_dir, transform = vt_transforms)
test_datasets = datasets.ImageFolder('./flowers/'+test_dir, transform = vt_transforms)
 Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True) #loss收敛稳定，加大batch_size
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32) 
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32) 




with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)





# TODO: Build and train your network
model = models.vgg16(pretrained=True)
model


# In[6]:


#替换分类器
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          #('drop1',nn.Dropout(0.5)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096,1000)),
                          #('drop2',nn.Dropout(0.5)),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(0.5)),
                          ('output',nn.Linear(1000,102)),
                          ('softmax',nn.LogSoftmax(dim=1))                     
                          ]))
    
model.classifier = classifier





print(model)





#Build and train your network

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
epochs = 5
print_every = 40
steps = 0

model.cuda()

for e in range(epochs):
    model.train() 
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() #trian loss
        
        if steps % print_every == 0:
            
            running_loss = 0
        #valid    
            valid_correct = 0
            valid_total = 0
            model.eval() 
            with torch.no_grad():
                for data in validloader:
                    images, valid_labels = data
                    valid_outputs = model(images.to('cuda'))
                    valid_loss = criterion(valid_outputs, valid_labels.to('cuda'))
                    _, predicted = torch.max(valid_outputs.data, 1)
                    valid_total += valid_labels.size(0)
                    valid_correct += (predicted.to('cuda') == valid_labels.to('cuda')).sum().item()
                    valid_loss += valid_loss.item()
                    
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Train_Loss: {:.4f}".format(running_loss/print_every),
                  "Valid_Loss: {:.4f}".format(valid_loss/len(validloader)),
                  'Valid Accuracy: %d %%' % (100 * valid_correct / valid_total))

            model.train()





#  Do validation on the test set
test_correct = 0
test_total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, test_labels = data
        test_outputs = model(images.to('cuda'))
        _, predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (predicted.to('cuda') == test_labels.to('cuda')).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * test_correct / test_total))


#  Save the checkpoint 
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [4096],
              'state_dict': model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'Epoch':5,
              'image_datasets':train_datasets.class_to_idx
             }

torch.save(checkpoint, 'checkpoint.pth')


#  Write a function that loads a checkpoint and rebuilds the model
from collections import OrderedDict
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096,1000)),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(0.5)),
                          ('output',nn.Linear(1000,102)),
                          ('softmax',nn.LogSoftmax(dim=1))                     
                          ]))
    
    model = models.vgg16(pretrained=True)
    model.classifier = classifier
    
    model.optimizer = checkpoint['optimizer'],
    model.epoch =  checkpoint['Epoch'],
    model.image_datasets = checkpoint['image_datasets']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# In[12]:


model = load_checkpoint('checkpoint.pth')
print(model)



# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256,256))#调整图像大小
    

    im = im.crop((16,16,240,240))
    
    np_img = np.array(im)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = np_img/256

    img = (np_img-mean)/std

    img = img.transpose((2,0,1)) #→ Tensor
    
    return img


x=process_image('./flowers/test/1/image_06743.jpg')
x


# In[21]:


im = Image.open('./flowers/test/1/image_06743.jpg')
np.size(im)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[23]:


imshow(torch.from_numpy(process_image('./flowers/test/2/image_05133.jpg')))





def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to('cuda')
    model.class_to_idx = train_datasets.class_to_idx
    
    im_tensor = torch.from_numpy(process_image(image_path)).float() #转换 PyTorch 张量
    im_tensor = im_tensor.unsqueeze(0) #增加一个维度
    
    im = im_tensor.view(1,3,224,224)
    im = im.type_as(torch.FloatTensor())
    im = im.to('cuda')
   
    with torch.no_grad():
        output = model(im) 
        output = F.log_softmax(output,dim=1) #输出对数形式
        output = torch.exp(output) #转换成正常的机率值
        probs,classes = torch.topk(output,topk)
    
    return probs,classes
    # TODO: Implement the code to predict the class from an image file
    


# In[35]:


probs, classes = predict('./flowers/test/2/image_05133.jpg', model,topk=5)
print(probs)
print(classes)





a = train_datasets.class_to_idx
idx_to_class={v: k for k, v in a.items()}#颠倒字典


# In[37]:


# \Display an image along with the top 5 classes
#im = process_image('./flowers/test/2/image_05133.jpg')
model.to('cuda')
model.class_to_idx = train_datasets.class_to_idx
probs,classes = predict('./flowers/test/2/image_05133.jpg', model,topk=5)

probs = probs.cpu().detach().numpy().tolist()[0] #从 Tensor 转换为 numpy 
classes=classes.cpu().detach().numpy().tolist()[0]

prob_max = probs[0]
#花卉名称

flower_name=[]
for i in classes:    
    flower_name.append(cat_to_name[idx_to_class[i]])       

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,12))
imshow(torch.from_numpy(process_image('./flowers/test/2/image_05133.jpg')),ax1)
fig.suptitle(flower_name[0])

ax2.set_xlim(0,prob_max+0.05)
ax2.set_ylabel('classes')
ax2.set_xlabel('probs')
#imshow(torch.from_numpy(process_image('./flowers/test/1/image_06743.jpg')))
ax2.barh(flower_name,probs,alpha=1)
plt.show






