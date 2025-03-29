# UNET_double applied to PLANET image containing 11 bands: 8 Planet images  + 3 indexes (NDVI, MSAVI2 and NDWI). The area os study is TILE 130 from the Arizona Grid of the Tree Stress Project
This is an Python code to apply the UNET model to a Planet image composite containing 8 Planet bands plus three vegetation indixes: NDVI, MSAVI2 and NDWI. Total 11 bands for TILE 130 of the Arizona tiles used for the Tree Stress Project.
The file "Define_UNET_Architecture.py" contains a two sections: 1) Training the UNET model and 2) Define the UNET Architecture.
  1) The training UNET model prepares the data set for training.
    a)   Split the dataset into tiles/patches: Split the satellite image and the mask images into patches. Deep learning models typically take 256*256 or 512*512 patches. Rasterio + numpy are used during this process.
    b)   Create train/validation/test datasets
    c)   Normalize the input images
  2)  Train a UNET model for multiple classes segmentation with:
      a) Input = 11 bandas images
      b)  Output: 5 classes segmenation mask
      c)  Loss function: categorical cross-Entropy
      d)  Activation: Softmax
  3)  Define the UNET Architecture for multiple classes
      a) Uses a softmax activation instead a sigmoid
  4)  Compile the model



![image](https://github.com/user-attachments/assets/9552f989-c7e8-4255-9372-c5d76cd2440a)


[Using U-Net-Like Deep Convolutional Neural Neural Networks for Precise Tree Recognition in very high resolution Satellite images.pdf](https://github.com/user-attachments/files/19522980/Using.U-Net-Like.Deep.Convolutional.Neural.Neural.Networks.for.Precise.Tree.Recognition.in.very.high.resolution.Satellite.images.pdf)
