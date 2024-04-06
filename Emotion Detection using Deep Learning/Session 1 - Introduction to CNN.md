# **Convolution neural networks CNN, convents**

1) **Assumptions**:
   1. Basic understanding of Python
   2. Basics of ML & DL
   3. (Matplotlib for plotting)

2) **Session to be Covered**:
   1. Complete theory & history of CNN
   2. Coding of CNN (cat vs dog)
   3. Transfer learning -> end to end project
   4. Hyper parameter tuning & performance metrics
   5. Deployment

### What are images and pixels

Images on your computer screen or a digital photograph are not exactly what they seem. They areを作って (つくられて,つくられる - tsukurarete, tsukuraれる - created) from millions of tiny building blocks called pixels.

**Pixels: The Building Blocks**

* A pixel, short for picture element, is the smallest individual unit of a digital image. 
* Imagine a mosaic – a picture made from many small tiles. Each tile in the mosaic is like a pixel in a digital image. 
* Millions or even billions of pixels come together to form the images you see on your screens.
* Each pixel itself can’t display a vast range of colors, but by combining them in different ways, they can create a whole spectrum of colors. 
* We don't want to have more than one colors in one pixel so we increase the number of pixels so that we can have one color per pixel to get more
* accurate and valuable information from given image.
* There are 0 to 255 values in a single pixel.

**How Pixels Work**

* Most digital screens use red, green, and blue (RGB) colors to create all the other colors you see. 
* Each pixel on the screen stores information about how much red, green, and blue light needs to be turned on to create a specific color.
* The more bits used to store this information, the more colors a single pixel can represent, and the more realistic the image will look.

**Resolution: How Many Pixels**

* The number of pixels in an image determines its resolution. Resolution is often given in width x height format, like 1920x1080. 
* The higher the resolution (more pixels), the sharper and more detailed the image will be.
* Low-resolution images, with fewer pixels, will appear grainy or blocky if you zoom in too close. This is because you’re seeing the individual pixels.

**Putting It All Together**

* By combining millions of colored pixels, we can create incredibly realistic and detailed digital images.
* From the photos on your phone to the videos you watch online, everything relies on these tiny building blocks. 

The visual cortex is a part of the brain responsible for processing the complex information sent to it from your eyes. Here's a breakdown of what it is and how it works:

**Where It's Located:**

* **Occipital Lobe:** The visual cortex covers a large part of the occipital lobe, the brain region located at the very back of your head.

**What It Does:**

* **Sees, Sorts, and Interprets:** It's not as simple as your eyes just transmitting a picture directly.  The visual cortex takes the raw data from your eyes and does the following:
    * **Assembles:** It puts together the different bits of information, like lines, colors, movement, shapes, etc.
    * **Analyzes:**  It breaks down the information to recognize objects, patterns, and depth.
    * **Gives Meaning:** It connects what you're seeing to your existing memories and experiences to understand what the object is and what it means (is that a ball, a threat, something delicious?).

**Not Just One Area:**

* **Sections:** The visual cortex has different subdivisions (called V1, V2, V3, etc.) that become more specialized as they move further and further away from the initial receiving area.
* **Two Pathways:** There seem to be two major pathways:
    * **"What" Pathway:** Focused on identifying objects.
    * **"Where" Pathway:** Deals with location and motion. 

**Damage to the Visual Cortex:**

* Damage to this area can lead to a variety of visual impairments. This can range from partial vision problems to complete blindness, even though the person's eyes are still working just fine. 
In Convolutional Neural Networks (CNNs), a kernel (also known as a filter) is a small, learnable matrix that plays a crucial role in feature extraction. Here's what you need to know:

**How Kernels Work**

1. **Sliding Window:** The kernel slides over the input data (usually an image) in a convolutional manner. Think of it like a magnifying glass moving across a map.
2. **Element-wise Multiplication and Summation:** At each position, the kernel performs element-wise multiplication between its values and the corresponding values in the input image. The products are then summed up to produce a single number.
3. **Feature Map:** As the kernel slides over the entire input, it creates a new, smaller output called a feature map (or activation map).

**What Kernels Do**

* **Feature Extraction:**  Different kernels are trained to detect specific features within the image. For example:
    * **Edge Detectors:** Kernels can be designed to identify horizontal, vertical, or diagonal edges.
    * **Blurring:** Kernels can smooth out an image by averaging pixel values in a neighborhood.
    * **Sharpening:**  Certain kernels enhance edges and details in an image.
    * **More Complex Features:**  In deeper CNN layers, kernels learn increasingly complex feature representations, from simple edges to textures and even object parts.

**Key Properties of Kernels**

* **Size:** They are usually small squares (e.g., 3x3, 5x5).
* **Stride:**  The stride determines how far the kernel jumps across the image with each step. A higher stride results in a smaller feature map.
* **Padding:** Sometimes padding with zeros is added around the image to control the output size.
* **Weights:** The values within the kernel are the learnable parameters. During the training of a CNN, these weights are optimized to extract the most relevant features for the task at hand.

**The Power of Kernels**

CNNs are powerful because they learn the best kernel weights automatically to identify the features that matter most for a task. This makes them incredibly effective at computer vision tasks like image classification, object detection, and many others.

**Let me know if you'd like a specific example of how a kernel works, or if you want to dive deeper into any particular aspect of CNNs!** 

The Sobel horizontal kernel (or Sobel operator) is a specific type of filter used in image processing for edge detection, primarily emphasizing horizontal edges. Here's how it works:

**What is Edge Detection?**

* Edge detection aims to find the boundaries within an image where there is a sharp change in brightness or color. These edges often signify the outlines of objects or transitions between different textures.

**The Sobel Operator**

* It's based on the idea of calculating the gradient of an image. The gradient measures the rate of change of pixel intensity in a particular direction.
* The Sobel operator uses two kernels: one for the horizontal gradient (Gx) and one for the vertical gradient (Gy). We'll focus on the horizontal one here.

**Sobel Horizontal Kernel (Gx)**

It's usually represented as a 3x3 matrix:

```
Gx =  [-1  0  1]
      [-2  0  2]
      [-1  0  1]
```

**How it works**

1. **Convolution:** The kernel is convolved with the image (just like the kernels we discussed in CNNs). This means it slides over the image, performs element-wise multiplication of the kernel's values with the corresponding image pixels, and sums up the results.
2. **Approximating the Horizontal Gradient:** This convolution operation gives a high value if a sharp horizontal change in intensity is present at the kernel's location. It highlights horizontal edges.  

**Why is it Useful?**

* **Sensitivity to Horizontal Edges:** It's specifically designed to emphasize edges that change significantly along the horizontal axis.
* **Computationally Simple:** It's easy to calculate, making it efficient for many image processing applications. 

**Remember:**

* The Sobel operator also has a vertical kernel (Gy) that works similarly but detects vertical edges.
* They are often used together to provide a more comprehensive edge detection result.

Absolutely!  Here's a breakdown of some of the most common and important types of kernels in computing and machine learning:

**1. Operating System Kernels**

* **The Core of an Operating System:** Kernels reside at the heart of operating systems (like Windows, macOS, Linux) and manage core functions like:
   * Resource allocation (memory, CPU time)
   * Device drivers
   * System calls and security

* **Types of OS Kernels:**
   * **Monolithic Kernels:**  Include most of the OS's core functionality in the kernel space. (Example: Linux kernel)
   [Image of Monolithic Kernel]
   * **Microkernels:** Have a smaller kernel with core functionalities and use modules to provide other OS services. (Example: GNU Hurd)
   [Image of Microkernel]
   * **Hybrid Kernels:**  Combine aspects of both monolithic and microkernels. (Examples: Windows NT, macOS XNU)
   [Image of Hybrid Kernel]

**2. Convolutional Neural Network (CNN) Kernels**

* **Feature Extractors:** Small matrices used to slide over images extracting specific features. (We discussed this earlier)
* **Examples:**
   * **Edge Detection:** Sobel, Prewitt, etc.
   [Image of Edge Detection Kernels]
   * **Sharpening Kernels**
   [Image of Sharpening Kernels]
   * **Blurring Kernels** 
   [Image of Blurring Kernels]

**3. Kernels in Support Vector Machines (SVMs)**

* **Transforming Data:**  Kernels in SVMs help map data into higher-dimensional spaces where it may become easier to separate into different classes. 
* **Types of SVM Kernels:**
   * **Linear Kernel:** For linearly separable data.
   [Image of Linear Kernel]
   * **Polynomial Kernel:** For non-linear relationships.
   [Image of Polynomial Kernel]
   * **Radial Basis Function (RBF) Kernel:** Very flexible and popular choice.
   [Image of Radial Basis Function Kernel]
   * **Sigmoid Kernel** 
   [Image of Sigmoid Kernel]

In Convolutional Neural Networks (CNNs), padding is a crucial technique used to manage the spatial dimensions (height and width) of data throughout the network. Here's a detailed explanation:

**Why Padding is Important:**

* **Preserves Information:** Without padding, the size of the output feature map typically shrinks after each convolution operation due to the filter "sliding" over the input. This can lead to information loss, especially at the edges of the image. Padding adds extra pixels around the border of the input, mitigating this issue.
* **Controls Output Size:** Padding allows you to control the size of the output feature map. This can be beneficial for maintaining consistent dimensions across certain network architectures or ensuring specific output sizes for tasks like image segmentation. 
* **Improves Training Stability:** Padding can also contribute to a more stable training process by maintaining similar receptive fields (the area of the input an output element "sees") across convolutional layers. 

**Types of Padding in CNNs:**

1. **Same Padding:**

   * In same padding, the goal is to maintain the same output height and width as the input.
   * The network calculates how much padding to add by considering the filter size, stride (movement of the filter), and desired output dimension. 
   * The padding amount (p) is usually calculated as: 
      p = (floor((filter size - 1) / 2)) 
   * Padding is added equally to all sides of the input feature map (top, bottom, left, right).

2. **Valid Padding (No Padding):**

   * In valid padding, no extra pixels are added to the input feature map. 
   * This results in a smaller output size compared to the input due to the filter "eating away" at the edges.
   * Valid padding might be useful when you want to gradually shrink the feature maps as you go deeper into the network to capture higher-level features or reduce computational cost. 

**Example: Consider a 3x3 filter with a stride of 1 applied to a 4x4 input image.**

* **Without padding (Valid Padding):** The output will be a 2x2 feature map, losing information from the edges.
* **With same padding (padding of 1):** The output will remain a 4x4 feature map, preserving all the information.

**Choosing the Right Padding:**

The choice between same padding and valid padding depends on your specific network architecture and task. Here's a general guideline:

* **Same padding:** Use when you want to preserve spatial information or maintain consistent feature map sizes across layers.
* **Valid padding:** Use when you want the feature maps to shrink as you go deeper (common in some architectures) or when computational cost is a concern.

**Additional Points:**

* Padding can be applied with different values (not just 1) depending on the desired output size and network design.
* In some cases, reflection padding or replication padding might be used, where the edge pixels of the image are mirrored or repeated to create the padding, respectively. These are less common than same and valid padding.


Strides in Convolutional Neural Networks (CNNs) are a hyperparameter that controls how the filter (kernel) moves over the input image or feature map. They have a significant impact on the network's output and its ability to capture features. Here's a breakdown of how strides work and why they matter:

**What is Stride?**

* **Step Size:** The stride determines the number of pixels the filter jumps as it slides over the input.
* **Example:**
   * A stride of 1 means the filter moves one pixel at a time.
   * A stride of 2 means it jumps two pixels with every move.

**Effect of Stride on Output Dimensions**

* **Downsizing:**  A stride greater than 1 reduces the spatial dimensions (height and width) of the output feature map compared to the input. This type of downsampling can:
    * **Reduce Computation:** Lower the number of calculations.
    * **Focus on Larger-Scale Features:** Make the network less sensitive to small shifts and variations in the image.

* **Calculation** Here's the formula to calculate the output feature map size with strides:

   Output Size = ((Input Size - Filter Size  + 2 * Padding) / Stride) + 1

**Common Practices and Considerations**

* **Stride of 1:** A stride of 1 is common for detailed feature extraction. The filter examines every pixel and adjacent pixels thoroughly.

* **Stride of 2 or More:** Stride values of 2 or more are used for downsampling, often in conjunction with pooling layers. This helps build deeper networks with a wider receptive field, focusing on global features.

* **Trade-offs:** Increasing the stride reduces the output size, which can lead to information loss. It's essential to find a balance between reducing computation and preserving spatial information.

**Stride vs. Pooling**

Both strides and pooling downsample feature maps. However:

* **Strides** do so within the convolution operation itself by skipping pixels.
* **Pooling** performs separate maximum or average downsampling after the convolution.

**Choosing the Right Stride**

The optimal stride depends on your specific CNN architecture and task. Here's a guideline:

* **Start with a stride of 1** for detailed feature maps.
* **Use larger strides combined with pooling** for downsampling if computational efficiency is essential and focusing on larger-scale features is appropriate.

 Pooling layers are essential components of Convolutional Neural Networks (CNNs), playing a crucial role in image processing and computer vision tasks. Let's dive into how they work and why they're important:

**What is Pooling?**

* **Downsampling:** Pooling layers reduce the dimensions of feature maps generated by convolutional layers. They do this by summarizing a region (or window) within the feature map into a single representative value.
* **Intuition:** It's like looking at a picture from further away – you lose some detail, but the overall features and shapes remain recognizable.

**Why use Pooling?**

1. **Computational Efficiency:** Reduces memory and computational load by decreasing the number of parameters in the network. This makes training faster.
2. **Translation Invariance:** Provides some robustness to small changes and shifts in the input image. If a feature is detected slightly differently in one part versus another, pooling helps make it count as the same feature.
3. **Controlling Overfitting:** Pooling helps prevent CNNs from overfitting by reducing the chance of them memorizing highly specific details of the training data.

**Common Pooling Types**

* **Max Pooling:** The most popular type. It takes the maximum value within the pooling window, highlighting the most prominent activation (feature detected) in that area.
* **Average Pooling:**  Calculates the average of all values within the pooling window. Used less frequently, but it can have a smoothing effect on the feature maps.
* **L2-Norm Pooling:** A less common variation that takes the square root of the sum of squares within the pooling window.

**How Pooling Works**

1. **Pooling Window:** A window size (e.g., 2x2) and a stride (how much that window moves) are defined.
2. **Application:** The window slides across the feature map, performing the pooling operation (Max or Average) at each position.
3. **Output:** A new, smaller feature map is generated. 

**Key Points**

* Pooling layers typically don't have learnable parameters. They use fixed operations (like taking max or average).
* Pooling layers are usually placed after convolutional layers and are often interspersed throughout the network architecture.

**Example:**

Consider a 4x4 feature map and a max pooling operation with a 2x2 window and a stride of 2:

```
Input:   1  5  3  8     Output:  5  10
         8  2  7  10            8  12  
         9  6  1  4
         3  10 12 2
```

1. only kernel -> (N-K+1) x (N-K+1)
2. Paddy + kernel -> (N-K+2P+1) x (N-K+2P+1)
3. strdy + kernel -> (N-K+S-1)/S x (N-K+S-1)/S
4. Paddy + strdy of kernl -> (N-K+2P+S-1)/S x (N-K+2P+S-1)/S



