# Computer Vision Assignment 2

> Authors (Group 2):
>
>  - Aditya Shourya (i6353515)
>
>  - Ashkan Karimi Saber (i6365010)
>
>    code at : https://github.com/adishourya/vit_fer2013



## Structure of the Report

* The focus of the report would not be on attaining the best test score but would be on experimenting with different architectures.
* And looking through how the forward pass looks in each case.
* And we will try to reason rate of learning  whenever applicable
* And finally we will pass in a pair of images as sampled from a video

### Convolutional Neural Network

* we will first develop a simple convolutional neural network
* And reason the effectiveness of the netwrok on a simple task as FER 2013[https://www.kaggle.com/datasets/msambare/fer2013]

### A simple hand made vision transformer (shallow)

* we will try to replicate the paper <u>"AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"</u> -- adosovitskiy et.al [https://arxiv.org/pdf/2010.11929v2] and show through one full forward pass.

### Pretrained Vision Transformer

* we will use vit_b_16 [https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#vit-b-16] and unfreeze the last few layers and attention heads to perform transfer learning.

  

  

## FER2013 Dataset

>The dataset comprises **48x48** pixel grayscale images of faces. These faces have been automatically aligned to ensure that each face is roughly centered and occupies a similar amount of space in every image.
>
>The objective is to classify each facial expression into one of seven emotional categories: 0 for Angry, 1 for Disgust, 2 for Fear, 3 for Happy, 4 for Sad, 5 for Surprise, and 6 for Neutral. The training set includes 28,709 examples, while the public test set contains 3,589 examples.
>
>​	--  [https://www.kaggle.com/datasets/msambare/fer2013]

<img src="/Users/adi/Library/Application Support/typora-user-images/image-20240522205713433.png" alt="image-20240522205713433" style="zoom:50%;" title = "Class Distribution in FER2013"/>

Class Diftribtion in FER2013, and the challenges asscociated with the dataset are :

>  The FER2013 dataset has several inherent issues that make it challenging for deep learning architectures to achieve optimal results. Key problems include ==imbalanced data,== intra-class variation, and occlusion. Specifically, the dataset exhibits significant imbalance in the training data, with classes having vastly different numbers of samples. For example, the 'happy' emotion has over 13,000 samples, whereas 'disgust' has only about 600 samples, as shown in the figure above.
>
> Intra-class variation refers to the differences within the same class. Reducing intra-class variation while increasing inter-class variation is crucial for effective classification. Variations, uncontrolled lighting conditions, and occlusions are common issues that face recognition systems encounter in real-world applications . These challenges often lead to a drop in accuracy compared to performance in controlled experimental settings. Occlusion occurs when an object blocks part of a person's face, such as a hand, hair, cap, or sunglasses. Although occlusion complicates face recognition, it can also provide useful information, as people often use their hands while communicating through gestures.
>
> ​		-- https://www.oaepublish.com/articles/ir.2021.16



## CNN

* All the experiments and the code are from the notebook : https://github.com/adishourya/VIT_FER2013/blob/main/convolutional_augmentation.ipynb
* Basic Augmentations applied: horizontal and vertical fliiping for our model to become invariant to geometrical transformations.
* We dont use any adjustment sharpness ; because we saw it made the lips and the expression overly smooth to discern emotion.

```python
self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomAdjustSharpness(sharpness_factor=2), # makes lips look worse and it performs slightly worse with it
                transforms.Resize((32,32), antialias=True)]) # limit number of patches (4) calculation.. keep this a multiple of 16*16

```

* A simple neural network from the documentaion in pytorch.

```python
# define a small convolutional network
# see beautiful mnist in tinygrad .
import torch.nn as nn
import torch.nn.functional as F

# shape after operations n,n ->(with padding p and stride s) (n + 2p - f + 1 )/s + 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # input channel 1 , 6 filter banks each of kernels size (5,5)
        self.pool = nn.MaxPool2d(2, 2) # this is not a learnable operaation just performs downsampling
        self.conv2 = nn.Conv2d(6, 16, 5) # 16 kernels 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7) # we have 7 classes 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


net = Net()
```

* we choose commonly used 5,5 filters in our case. with padding of 0 and stride of 1.

### forward pass in our learnt convolutional layers

```python
# lets visualize the first layer on a sample image
x = self.pool(F.relu(self.conv1(x)))

# 1,6 input channel = 1 (greyscale)
# output dimension = 6 (filterbanks)
# kernel size = (5,5) kernels
net.conv1 , net.conv1.weight.shape

# > out: (Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)), torch.Size([6, 1, 5, 5]))
```



<img src="/Users/adi/Library/Application Support/typora-user-images/image-20240524224750128.png" alt="image-20240524224750128" style="zoom:50%;" />

```python
import torch.nn.functional as F
with torch.no_grad():
    # first layer convolution
    out1 =F.conv2d(sample_img,net.conv1.weight,
                   bias=None, stride=1, padding=0)
    print(out1.shape) # input channel = 1 filter banks = 6
    
    
plt.imshow(
    einops.rearrange(out1,"out_c h w -> h (out_c w)"),
    cmap="grey"
)
# visualizing the output of convolution from the first learnt layer.
# notice how some kernels are vastly different
```

![image-20240524225027213](/Users/adi/Library/Application Support/typora-user-images/image-20240524225027213.png)

* we will not show the pooling operation here (check notebook) as it is not a learnable parameter

```python
with torch.no_grad():
    # first layer convolution
    out2 =F.conv2d(out1_p,net.conv2.weight,
                   bias=None, stride=1, padding=0)
    print(out2.shape) # input channel = 1 filter banks = 6

plt.figure(figsize=(15,8))
plt.imshow(
    einops.rearrange(out2,"out_c h w -> h (out_c w)"),
    cmap="grey"
)
```



![image-20240524225239099](/Users/adi/Library/Application Support/typora-user-images/image-20240524225239099.png)

* and then we maxpool it and then flatten it to pass to a feed forward network to arrive at the logits for classification

* we only train for 10 epochs as attaining the best test accuracy is not our main goal of this report

  ![image-20240524225805968](/Users/adi/Library/Application Support/typora-user-images/image-20240524225805968.png)

![image-20240524225828370](/Users/adi/Library/Application Support/typora-user-images/image-20240524225828370.png)

```
Test Accuracy of Classes (Generalization scores of our Convolutional Model)

Angry	: 10% 	 (54/491)
Disgust	: 0% 	 (0/55)
Fear	: 7% 	 (41/528)
Happy	: 80% 	 (710/879)
Sad	: 32% 	 (191/594)
Surprise	: 47% 	 (197/416)
Neutral	: 33% 	 (208/626)

Test Accuracy of Dataset: 	 39% 	 (1401/3589)

```







## Shallow Transformer

* we will try to replicate a shallow vit from the paper  adosovitskiy et.al [https://arxiv.org/pdf/2010.11929v2] 
* `AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE`

<img src="/Users/adi/Library/Application Support/typora-user-images/image-20240524191905674.png" alt="image-20240524191905674" style="zoom:100%;" />

* All the experiments and the code are from : https://github.com/adishourya/VIT_FER2013/blob/main/fer_vit.ipynb

* we will first start with splitting the image as patches

### generating patches

> To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens
>
> ​	-- Introduction  adosovitskiy et.al

> Naive application of self-attention to images would require that each pixel attends to every other pixel. 
>
> ​		--  Related Work  adosovitskiy et.al

```python
## from class FerDataset(utils.data.Dataset): (check Notebook fer_vit)
  def get_patches(self, arr):
        # best place to use einops
        # squueze out the channel dimension.. the loader will add it ..
        arr = einops.rearrange(arr,"1 h w -> h w") 
        patches = einops.rearrange(
            arr, "(p1 h) (p2 w) ->(p1 p2) h w",p1=2 , p2=2
        ) # lay it on as the batch for the image
        # we need to crop twice for a 48 * 48 image to get a 16 * 16 patch
        return patches

```

![image-20240525001520248](/Users/adi/Library/Application Support/typora-user-images/image-20240525001520248.png)

### Embed Patches

* We first flatten all the patches and embed them with a lookup table

```python
# initalize a lookup table
 self.C = nn.Parameter(torch.randn(16*16, self.n_embed) * 1/torch.sqrt(256) )
  
# flatten the patches
patches = einops.rearrange(X,"b p h w -> b p (h w)")

# embed them : the block matrix multiplication looks like :
# B , p_num , 256 @ 256 , 32
emb = patches @ self.C # B , p_num , n_embed
# so each patch (a matrix) gets and n_embed dimensional representation (row vector)
```



### class token

* we dont actually use the class token in our implemented model. check screenshot of the issue : `` Is the extra class embedding important to predict the results, why not simply use feature maps to predict?` in the appendix.
* But the pretrained models implement them anyway. So we do it like :

```python
# its like adding an extra patch . but this patch is learnable
learnable_patch = torch.randn(1,n_embed)
# add same learnable patch to all the images in the batch
learnable_patch = einops.repeat(learnable_patch,"1 n_embed -> 3 1 n_embed")
print(learnable_patch.shape)
# prints torch.Size([3, 1, 128]) # batch_size , 1 token , 128 embed dimension

# you actually need all dims except one to be different to pack or concat it
# in the paper they add it at the first location. (dont really see how it matters if append it at the end instead)
xb ,ps = einops.pack([learnable_patch,xb],"b * n_embed") # we want to append it on top of patch
xb.shape
# prints torch.Size([3, 5, 128]) # batch , tokens , n_embed

# 4 patches from the images , 1 patch from class token
```

### positional embedding

```python
# positional embedding
# the paper says Position embeddings are added to the patch embeddings to retain positional information.
# We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings

# init a patch embedding
self.pe = nn.Parameter(torch.randn(1,4,self.n_embed)) # each pach and representation should get positional embedding

# add patch embedding after class token
emb = emb + self.pe # kind of acts like a bias towards each patches.
```

### pass through transformer blocks

* We will first start with making self attention blocks. and then concatenate them to get multihead attention (vasawani et.al  2017) .
* Note multihead attention blocks returns back the same size of representation `n_embed` .
* unit test for our implementation of self head attention is in the appendix

```python
# check appendix for unit test of this class
class SingleHead(nn.Module):
    """
    Implements a single head of attention (unmasked)
    """

    def __init__(self,n_embed=32,head_size=8):
        super().__init__()
        # single head
        self.head_size = torch.tensor(head_size)
        self.n_embed = torch.tensor(n_embed)
        self.Q = nn.Parameter( torch.randn(self.n_embed,head_size) * (1/torch.tensor(2.82)))
        self.K = nn.Parameter( torch.randn(self.n_embed,head_size) * (1/torch.tensor(2.82)))
        self.V = nn.Parameter( torch.randn(self.n_embed,head_size) * (1/torch.tensor(2.82)))

    def forward(self,x):
        query = x @ self.Q
        key =   x @ self.K
        value=  x @ self.V
        
        # hand implementation
        # scale => sqrt head size
        scale = 1 / torch.sqrt(self.head_size)

        # we will not use any masking here as its an image
        # and no dropout consideration in this implementation
        comm = query @ key.transpose(-2,-1)
        comm = comm* scale
        soft_comm = torch.softmax(comm, dim=2)
        att = soft_comm @ value
        
        return att
```

```python
class Multihead(nn.Module):
    def __init__(self,n_embed,n_heads):
        super().__init__()
        
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_size = self.n_embed // self.n_heads
        
 				self.multiheads = nn.ModuleList(
            [SingleHead(self.n_embed,self.head_size)
             for _ in range(self.n_heads)]
        )

    def forward(self,x):
        return torch.cat([head(x) for head in self.multiheads],dim=2)
```

* And then a transformer block with it

```python
# only multihead ->  skip connection -> layernorm
# Batch norm : couples examples in and normalizes it .. (also has a regularization effect) but we need to keep a running mean to track new mean and sigma
# layernorm : normalizes the features of each example (does not couple examples across the batch) more popular in transformers

class TranformerBlock(nn.Module):
    def __init__(self, n_embed,n_head):
        super().__init__()
        self.multi_head = Multihead(n_embed,n_head)
        # i am not going to implement my own layer norm it wont be efficient and will be janky at best
        self.norm = nn.LayerNorm(n_embed) # we want to normalize feeatures (each patch gets normalized)
    
    def forward(self,x):
        # pass through multihead
        attention = self.multi_head(x)
        # skip connection and non linarity
        attention = torch.relu( x + attention)
        # layer norm
        attention = self.norm(attention)
        return attention # B , n_patch , n_embed
        ...
        
```

* So our architecure looks like this : 

```python
# most of the comments are pasted verbatim from the paper
class SmallVIT(nn.Module):
    
    def __init__(self):
        super().__init__()
        # patches
        # embedding
        self.vocab_size = torch.tensor(256) # 0 to 255 pixels
        # each patch will only get one n_embed representation
        self.n_embed = 32 # we will project each patch 16*16 to a 32 dimensional representation
        # so the lookup table would be of the shape
        # unlike in nlp where we embed token to a vector like below we would project matrix of patch size to a vector
        # self.C = nn.Embedding(self.vocab_size, self.n_embed)
        
        self.C = nn.Parameter(torch.randn(self.vocab_size, self.n_embed) * 1/torch.sqrt(self.vocab_size) )
        
        # positional embedding
        # the paper says Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings 
        self.pe = nn.Parameter(torch.randn(1,4,self.n_embed)) # each pach and representation should get positional embedding
        
        # we use the standard approach of adding an extra learnable “classification token” to the sequence
        self.classification_token = nn.Parameter(torch.randn(1, 1, self.n_embed))
        # we will keep the step above optional .. i dont understand why we should use it yet.
        
        # transformer block
        self.n_heads = 4 # we will use 4 heads for now
        self.transformer_block = TranformerBlock(self.n_embed,self.n_heads)
        
        # MLP Head for final logit calculation
        # n_patch * n_embed -> fer["emotion"].nuinque() : 7
        self.mlp_head = nn.Parameter(torch.randn(4*32, 7) * torch.sqrt(torch.tensor(4*32)))
        
        
        
        
        ...
       
    def forward(self,X):
        batch_size = X.shape[0]
        # flatten the patches
        patches = einops.rearrange(X,"b p h w -> b p (h w)")
        # B , p_num , 256 @ 256 , 32
        emb = patches @ self.C # B , p_num , n_embed
        emb = emb + self.pe # kind of acts like a bias towards each patches.
        
        # 2 transformers
        tf = self.transformer_block(emb)
        tf = self.transformer_block(tf)
        
        # flatten it : across patches
        tf = tf.view(batch_size,-1)
        
        # logits
        logits = tf @ self.mlp_head
        
        
        

        # broadcasting steps in the above command
        # B,p_n , p*p , n_embed
        #1,4,32


        return logits
```

![image-20240525004554044](/Users/adi/Library/Application Support/typora-user-images/image-20240525004554044.png)

![image-20240525004620165](/Users/adi/Library/Application Support/typora-user-images/image-20240525004620165.png)

```
Test Accuracy of Classes

Angry	: 12% 	 (61/491)
Disgust	: 0% 	 (0/55)
Fear	: 31% 	 (168/528)
Happy	: 62% 	 (551/879)
Sad	: 11% 	 (68/594)
Surprise	: 13% 	 (56/416)
Neutral	: 10% 	 (64/626)

Test Accuracy of Dataset: 	 26% 	 (968/3589)
```

* Note how the accuracy of a shallow transformer (significantly higher number of parameters compared to CNN) produces worse result. which is to be expected.
* This looks like its improving but even after 10 epochs the model is still ==confidently wrong.==

```python
# if we were to assume the model is predicting randomly from a uniform distribution
# i.e logits were rougly equal for all classes, then the loss would have been :
-1 * np.log(1/7) # which is 1.946. which is still lower than what we have after 10 epochs!
```

* note how other models (both cnn and pre-trained vit) already has a loss score of < 1.946 after first epoch.
* This implies that the ==model was not initalized well== , and since the model plateaus around the half way . it also indicates that the model ==does not have enough representational capacity.==



## Pretrained Vision Transformer (perform transfer learning)

* The code and experiments for this section can be found at : https://github.com/adishourya/VIT_FER2013/blob/main/vit_pretrained.ipynb
* we will be transfer learning from the model : vit_b_16 [https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#vit-b-16] and unfreeze the last few layers to perform the training.

```python
vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
```

```python
vision_transformer.heads

#Sequential(
#  (head): Linear(in_features=768, out_features=1000, bias=True)
#)
```

### Fine Tune

```python
# fine-tune with dataset

# change the number of output classes
vision_transformer.heads = nn.Linear(in_features=256*3, out_features=7, bias=True)

# freeze the parameters except the last linear layer
#
# freeze weights
for p in vision_transformer.parameters():
    p.requires_grad = False

# unfreeze weights of classification head to train
for p in vision_transformer.heads.parameters():
    p.requires_grad = True
```

* and then train

![image-20240525005716704](/Users/adi/Library/Application Support/typora-user-images/image-20240525005716704.png)



![image-20240524201652452](/Users/adi/Library/Application Support/typora-user-images/image-20240524201652452.png)







```
## Results for each class
Disgust	: 1% 	 (1/55)
Fear	: 29% 	 (155/528)
Happy	: 78% 	 (688/879)
Sad	: 35% 	 (211/594)
Surprise	: 62% 	 (260/416)
Neutral	: 54% 	 (344/626)

Test Accuracy of Dataset: 	 50% 	 (1799/3589)
```

* Note how its considerably better than shallow transformer that we made
* It also performs better than our small convolutional network (Note that the CNN has extremely lower number of parameters compared to vit_b16 )



## Testing on images sampled from a real video

* we test the images on our CNN model.

![image-20240526123537081](/Users/adi/Library/Application Support/typora-user-images/image-20240526123537081.png)

* transform the image as expected by the model

```python
# convert to grey scale
our_img= einops.reduce(our_img,"h w c -> h w","min")

# crop image into batch size of 2
# 2 batches
ims = einops.rearrange(our_img," h (p w) -> p h w",p=2)

# resize to the same shape as the training data and unsquueze the channel dimension
ims=  transforms.Resize((32,32), antialias=True)(ims)
ims = einops.rearrange(ims,"b h w -> b 1 h w")

# forward pass through our network.
with torch.no_grad():
    logits = net(ims)# logits
    print("logits",logits)
    probs = torch.softmax(logits,dim=1)
    print("probs",probs)
    print(torch.argmax(probs,dim=1))
```

```
logits tensor([[   8.8504, -107.3212,   49.5044,  -21.4002,  -39.1525,   90.2567,
          -52.2730],
        [  25.8041, -122.4651,   48.1215,  -10.1948,  -35.4007,   67.4436,
          -51.2039]])
probs tensor([[4.4226e-36, 0.0000e+00, 2.0023e-18, 0.0000e+00, 0.0000e+00, 1.0000e+00,
         0.0000e+00],
        [8.2455e-19, 0.0000e+00, 4.0601e-09, 1.9146e-34, 2.8026e-45, 1.0000e+00,
         0.0000e+00]])
tensor([5, 5])

classes[5] -> "surprise" # they were both classified as surprise.
# which is only partially correct . correct for the second image
```





## Remarks

* Number of parameters:

  * CNN uses significantly lower number of parameters because of the inherent nature of parameter sharing in convolutional neural network. And performs comparably to a vision transformer ==that's not been trained for a lot of epochs.==
  * when given enough scale , attention blocks could start learing context like the structural pattern as the filter banks do in convolutional networks.

* Discuss difficulty of task at hand

  * The user should use a simple convolutional network when given a task with less sample size.

  * > Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data. -- adosovitskiy et.al

    > However, the picture changes if the models are trained on larger datasets ==(14M-300M images).== We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks.
    >
    > ​	-- - adosovitskiy et.al



## Appendix

### unit test for self attention with pytorch's implementation

```python
def single_head(query, key,value):
    head_size = torch.tensor(query.shape[-1])
    # hand implementation
    # scale => sqrt head size
    scale = 1 / torch.sqrt(head_size)

    # we will not use any masking here as its an image
    # and no dropout consideration in this implementation
    comm = query @ key.transpose(-2,-1)
    comm = comm* scale
    soft_comm = torch.softmax(comm, dim=2)
    att = soft_comm @ value
    print(att.shape)
    return att


g=torch.Generator().manual_seed(123)
query, key, value = torch.randn(2, 3, 8 , generator = g), torch.randn(2, 3, 8, generator = g), torch.randn(2, 3, 8 , generator = g)

# our implementation
sh  = single_head(query,key,value)

# pytorch implementation
py_sa = nn.functional.scaled_dot_product_attention(query, key, value)

# > torch.allclose(py_sa , sh) prints True

```

### importance of class tokens ?	

Track issue at : https://github.com/google-research/vision_transformer/issues/61

![image-20240525002329076](/Users/adi/Library/Application Support/typora-user-images/image-20240525002329076.png)

* convolutional network without augmentation (which performs better ) is at : https://github.com/adishourya/VIT_FER2013/blob/main/convolutional.ipynb

## Refernces

* Dataset used for experiment : FER 2013[https://www.kaggle.com/datasets/msambare/fer2013]
* Tried to reproduce the paper :  `AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE` -- adosovitskiy et.al [https://arxiv.org/pdf/2010.11929v2
* Performed Transfer learning with :  vit_b_16 [https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#vit-b-16] 
