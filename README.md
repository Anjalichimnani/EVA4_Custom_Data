# Custom Data Creation with Depth Mask Estimation

**Office Data set** represents Techpark and office images with people walking on phone, strolling, in cafeteria and meeting conferences. The images constitute background images, transparent foreground images, foreground images flipped and foreground images overlaid on background in 20 different variations. There are masks created for all the foreground images and thus, all foreground background images.

Kinds of images (fg, bg, fg_bg, masks, depth)
Total images of each kind
The total size of the dataset
Mean/STD values for your fg_bg, masks and depth images

* Background Images:
  * [Link for Access] (https://github.com/Anjalichimnani/EVA4_Custom_Data/tree/master/bg_images)
  * Number of Images: 100
  * Dimensions: 224 X 224
  * Type: JPEG
  * Total Size: 2.64 MB

  ![BG_Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/bg_images.png)

* Foreground Images:
  * [Link for Access](https://github.com/Anjalichimnani/EVA4_Custom_Data/tree/master/fg_images)
  * Number of Images: 100
  * Dimensions: 112 X 112
  * Type: PNG
  * Total Size: 1.85 MB

  ![FG_Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/fg_images.png)

* Flipped Foreground Images:
  * Number of Images: 100
  * Dimensions: 112 X 112
  * Type: PNG
  * Total Size: 1.85 MB

  ![Flipped_FG_Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/fg_images_flip.png)

* Masks:
  * [Link for Access] (https://github.com/Anjalichimnani/EVA4_Custom_Data/tree/master/mask_images)
  * Number of Images: 400,000
  * Dimensions: 224 X 224
  * Type: JPEG
  * Total Size: 467 KB

  ![Mask_Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/mask_images.png)

* Foreground Background Images:
  * Number of Images: 400,000
  * Dimensions: 224 X 224
  * Type: JPEG
  * Total Size: 8.66 GB

  ![FG_BG_Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/fg_bg_images.png)

* Depth Masks:
  * Number of Images: 400,000
  * Dimensions: 224 X 224
  * Type: JPEG


The foreground images are transparent and are created using GIMP Tool.
The images are made transparent by adding alpha channel and removing the background by selecting the image foreground (poeple)

The Foreground images are flipped using the following code:
```

def flip_image_alpha (basepath, newpath):
    file_names = [entry for entry in os.scandir(basepath) if entry.is_file()]

    for idx, file_name in enumerate (file_names, start=1):
        image = cv2.imread (file_name.path, cv2.IMREAD_UNCHANGED)
        alpha = image[:,:,3]
        image = image[:,:,:3]

        image = cv2.flip (image, 1)
        alpha = cv2.flip (alpha, 1)
        result = np.dstack ([image, alpha])

        cv2.imwrite (newpath + file_name.name, result)

```

The Foreground background images are created adding each foreground image at 20 different location on the back ground image. The background image is divided into 28 X 28 blocks. Thus, placing the first pixel of the image at different 28 X 28 blocks (8 Blocks). The foreground images are ooverlaid ensuring the transparency is preserved. The code below helps to achieve the same:

```

def overlay_images (bgpath, fgpath, newpath, name_prefix, bg_img_size, fg_img_size):
    bg_file_names = [entry for entry in os.scandir(bgpath) if entry.is_file()]
    fg_file_names = [entry for entry in os.scandir(fgpath) if entry.is_file()]

    for fg_images in fg_file_names:
        for idx, bg_images in enumerate (bg_file_names, start=1):
            new_file_name = newpath + name_prefix + '_fg_' + fg_images.name[:-4] + '_bg_' + bg_images.name[:-4]

            bg_image = Image.open (bg_images.path)

            fg_image = Image.open (fg_images.path)

            x = 1
            for j in range (4, 0, -1):
                for i in range (0, 5):
                    fg_bg_image = bg_image.copy()
                    fg_bg_image.paste(fg_image, (28*i, 28*j), mask=fg_image)
                    fg_bg_image.save (new_file_name + '_{seq}.{suffix}'.format(seq=str(x).zfill(2), suffix = bg_images.name[-3:]))
                    x += 1

```

The Masks of the foreground images are created using the GIMP Tool by adding the alpha channel, selecting the image, masking the image and cropping it to desired size. The mask is then placed on back background as the foreground image on other backgrounds using the same code with addon image as background.

### Statistics:
* Foreground Background images:
  * Mean: [0.562814  0.5644937 0.5667319]
  * Standard Deviation: [0.23423661 0.22496194 0.24435855]

The code used to obtain the Mean and Standard Deviation for the complete records in dataset loader:
```
images = iter(dataset_loader).next()
numpy_images = images.numpy()

pop_channel_std = np.std(numpy_images, axis=(0, 2, 3))
pop_channel_mean = np.mean(numpy_images, axis=(0, 2, 3))

print (pop_channel_mean)
print (pop_channel_std)

```

### Depth Mask Estimation
The Depth Mask is created using the Reference: [Depth Mask](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb)
The model is loaded as below:
```
model = load_model('/content/DenseDepth/nyu.h5', custom_objects=custom_objects, compile=False)
```

The images are given as input as below:
```
inputs = load_images( glob.glob('/content/EVA4_Custom_Data/fp_bg_images/*.jpg') )
```

The outputs are consequently depicted as below:
```
outputs = predict(model, inputs)
```

## References:
[Depth Mask Estimation](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb)

[Custom Pytorch Dataloaders](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html?highlight=custom%20dataset)

[Mean and Standard Deviation](https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6)
