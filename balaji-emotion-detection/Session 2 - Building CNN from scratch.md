# Session 2 - Building CNN from scratch
Activation functions are an essential component of artificial neural networks. They introduce non-linearity into the network, allowing it to learn complex patterns and relationships in the data. 

Here's a detailed explanation of activation functions along with some commonly used examples:

**Understanding Activation Functions**

In an artificial neural network, a neuron receives weighted inputs from other neurons, sums them up, and then applies an activation function to determine its output. This output can then be fed to other neurons in the network. The activation function plays a crucial role in determining how the neuron transforms the input signal.

**Importance of Non-linearity**

Without activation functions, neural networks would only be able to learn linear relationships between the input and output. This is because stacking multiple linear layers results in another linear layer. Activation functions introduce non-linearity, allowing the network to learn more complex patterns and make non-linear decisions.

**Types of Activation Functions**

There are many different activation functions used in neural networks, each with its own properties and advantages. Here are a few commonly used examples:

* **Sigmoid Function:** This function squashes the input value between 0 and 1. It was a popular choice in earlier neural networks, but it can suffer from vanishing gradients during backpropagation, making it difficult to train deep networks.

* **ReLU (Rectified Linear Unit):** This function outputs the input directly if it's positive, otherwise it outputs zero. ReLU is a popular choice for many neural networks due to its computational efficiency and ability to alleviate the vanishing gradient problem.

* **Leaky ReLU:** This is a variant of ReLU that introduces a small slope for negative inputs. This helps to prevent the dying neuron problem, where ReLU neurons can become permanently inactive if they receive negative inputs for extended periods.

* **TanH (Hyperbolic Tangent):** This function squashes the input value between -1 and 1. It shares some properties with the sigmoid function but has a steeper slope, allowing for faster learning in some cases.

**Choosing the Right Activation Function**

The choice of activation function can affect the performance of your neural network. Here are some factors to consider when choosing an activation function:

* **Task:** Different tasks may benefit from different activation functions. For example, sigmoid functions are often used in output layers for classification tasks where the output represents a probability, while ReLU is a good choice for hidden layers in many applications.
* **Computational Efficiency:** Some activation functions, such as ReLU, are computationally more efficient than others, such as sigmoid. This can be an important factor for large neural networks.

Activation functions are a core component of neural networks. They determine the output of a neural network model, the complexity it can capture, and how well it can generalize. The purpose of an activation function is to introduce non-linearity into the output of a neuron. This is important because most real-world data is non-linear and we want neural networks to capture such data patterns.

### How Activation Functions Work

In a neural network, each neuron receives input from some other neurons or from an external source. The neuron then combines the input linearly using weights and biases (a process known as linear transformation). The activation function is applied to this linear combination to produce the neuron's output. The choice of activation function directly influences the ability of the neural network to converge and the speed at which it converges to a solution.

### Types of Activation Functions

1. **Sigmoid or Logistic Activation Function**: It squashes the input values into a range between 0 and 1. It is historically popular but has fallen out of favor due to issues like vanishing gradients.

```python
9b934b5f-44c3-4906-a5d2-4038ac7e864f

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

```

2. **Tanh (Hyperbolic Tangent) Activation Function**: It outputs values between -1 and 1. It is similar to the sigmoid but improved in terms of the range of values, reducing the risk of vanishing gradients to some extent.

```python
def tanh(x):
    return np.tanh(x)
```

3. **ReLU (Rectified Linear Unit) Activation Function**: It outputs the input directly if it is positive; otherwise, it will output zero. It has become very popular due to its computational efficiency and because it allows models to converge faster while reducing the likelihood of vanishing gradients.

```python
def relu(x):
    return np.maximum(0, x)
```

4. **Leaky ReLU**: It is a variant of ReLU designed to allow a small, non-zero gradient when the input is negative, which helps to keep the gradient flow alive during the training process.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)
```

5. **Softmax**: Often used in the output layer of a classifier, it converts logits to probabilities by taking the exponential of each output and then normalizing these values by dividing by the sum of all the exponentials.

```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
```

### Choosing an Activation Function

The choice of activation function depends on the specific application and the problem you are trying to solve. For example:

- **Sigmoid** and **tanh** are often used in layers of a neural network where you need the output to be normalized.
- **ReLU** and its variants (like Leaky ReLU) are very popular for hidden layers because they help with faster convergence and reduce the problem of vanishing gradients.
- **Softmax** is typically used in the output layer for multi-class classification problems.

In practice, the choice of activation function can significantly affect the performance of your neural network, and it's often beneficial to experiment with different functions to see which works best for your specific dataset and problem.


Kernel initialization and bias are crucial concepts in the context of neural networks, particularly when setting up the layers of a neural network model. These concepts play a significant role in the convergence and performance of the model.

### Kernel Initialization

Kernel initialization, also known as weight initialization, refers to the method used to initially set the weights of the layers in a neural network before training begins. Proper initialization is vital for ensuring that the network converges efficiently during training. If the weights are not initialized correctly, it could lead to issues such as slow convergence or even prevent the network from converging at all.

There are several strategies for weight initialization, including:

1. **Random Initialization**: Weights are initialized to small random values. This can be further refined to methods like uniform or normal distribution within a specific range.
   
2. **Xavier/Glorot Initialization**: This method adjusts the scale of the initialization based on the number of input and output neurons, aiming to keep the variance of activations consistent across layers. It's particularly useful for networks using sigmoid or tanh activation functions.

3. **He Initialization**: Similar to Xavier initialization but designed for layers with ReLU activation functions. It considers the non-linearity of ReLU to adjust the scale of weight initialization.

### Bias Initialization

Bias initialization refers to the process of setting the initial values of the bias terms in neural network layers. Bias terms are added to the weighted sum before the activation function is applied, allowing the activation function to be shifted to the left or right, which can help the model learn patterns more effectively.

Common strategies for bias initialization include:

1. **Zero Initialization**: Initializing all bias terms to zero. This is often a safe default and works well in many scenarios, especially when combined with proper weight initialization.

2. **Small Random Values**: Similar to weight initialization, biases can also be initialized to small random values, although this is less common.

3. **Constant Value**: In some cases, initializing biases to a small constant value other than zero (e.g., 0.01) can help avoid initial dead neurons in the case of ReLU activation functions.

### Importance

The choice of initialization can have a significant impact on the learning dynamics of a neural network. Proper initialization helps in achieving faster convergence and improving the overall performance of the model. It's often beneficial to experiment with different initialization strategies to find the one that works best for your specific architecture and problem.

In practice, modern deep learning frameworks provide built-in functions for both kernel and bias initialization, allowing for easy experimentation and implementation of the strategies mentioned above.



## CNN Architecture and Kernels Explained

Convolutional Neural Networks (CNNs) are a powerful type of neural network architecture specifically designed for image recognition and analysis tasks. They excel at capturing spatial features and relationships within images. Let's break down the key components of CNN architecture and how kernels play a vital role:

**CNN Architecture Breakdown:**

A typical CNN architecture consists of several building blocks stacked together in a specific order. Here's a breakdown of the essential components:

1. **Convolutional Layer:** This is the core building block of a CNN. It applies a filter, also known as a kernel, to the input image. The kernel slides across the image, performing element-wise multiplication (dot product) between its weights and a small region of the input image. This operation captures local features like edges, lines, and shapes. Multiple kernels can be used in a single convolutional layer, detecting various features.

2. **Activation Layer:** Following the convolutional layer, an activation function is applied to introduce non-linearity into the network. This is crucial because it allows the network to learn complex patterns beyond simple linear relationships. Common activation functions used in CNNs include ReLU (Rectified Linear Unit) and Leaky ReLU.

3. **Pooling Layer (Optional):** This layer performs downsampling to reduce the dimensionality of the data, making the network more computationally efficient and less prone to overfitting. Pooling operations like max pooling select the maximum value from a predefined window within the feature map.

4. **Fully Connected Layer:** After processing by convolutional and pooling layers, the data is flattened into a vector and fed into fully connected layers, similar to traditional neural networks. These layers perform classification or regression tasks based on the extracted features.


**Kernels: The Heart of Feature Extraction**

* **What is a Kernel?:** A kernel, also called a filter, is a small square or cube-shaped matrix of learnable weights. It essentially acts as a feature detector that slides across the input image, extracting specific features based on the pattern of weights within the kernel.

* **How Kernels Work:** During the convolutional operation, the kernel element-wise multiplies the corresponding elements in a local region of the input image. The results are then summed up, and this value becomes the output for that specific location in the feature map. The kernel then slides over one position and repeats the process, generating a feature map that highlights the presence of the features the kernel was designed to detect.

* **Multiple Kernels:** A convolutional layer can have multiple kernels, each with different weight patterns. This allows the network to learn and detect a variety of features within the image. For example, one kernel might detect horizontal edges, another vertical edges, and others might detect corners or specific shapes.

* **Kernel Size and Learning:** The size of the kernel (e.g., 3x3, 5x5) determines the extent of the local region it analyzes within the image. The weights within the kernel are learned during the training process, allowing the network to optimize them for detecting the desired features.


**Understanding CNNs with Kernels helps you grasp how these powerful networks process images. By applying multiple convolutional layers with various kernels, CNNs can progressively extract higher-level features, ultimately leading to accurate image recognition and analysis.**

Training a Convolutional Neural Network (CNN) involves several steps and considerations to ensure that the model learns effectively from the input data. Below is a high-level overview of the process to train a CNN model, typically used for tasks like image classification, object detection, etc.

### 1. Prepare Your Dataset

- **Collect Data**: Gather a sufficiently large and relevant dataset. For image tasks, this means collecting various images that represent all the classes you want the model to recognize.
- **Preprocess Data**: Normalize the images (e.g., scale pixel values to the range [0, 1] or [-1, 1]), resize them to a uniform size, and apply data augmentation techniques to increase the diversity of the training set.
- **Split Data**: Divide your dataset into training, validation, and test sets. A common split ratio is 80% for training, 10% for validation, and 10% for testing.

### 2. Define the CNN Architecture

- **Input Layer**: This layer should match the shape of your preprocessed images.
- **Convolutional Layers**: Apply convolutional layers to extract features from the images. You may include pooling layers to reduce dimensionality.
- **Activation Functions**: Use activation functions like ReLU in the convolutional layers to introduce non-linearity.
- **Fully Connected Layers**: After several convolutional and pooling layers, add one or more fully connected layers to perform classification based on the extracted features.
- **Output Layer**: The final layer should have as many neurons as there are classes, with a softmax activation function for multi-class classification tasks.

### 3. Compile the Model

- **Choose an Optimizer**: Common choices include Adam, SGD, and RMSprop. The optimizer will adjust the weights of the network during training to minimize the loss.
- **Loss Function**: For multi-class classification, `categorical_crossentropy` is commonly used. For binary classification, `binary_crossentropy` is a typical choice.
- **Metrics**: Specify metrics to monitor during training, such as accuracy.

### 4. Train the Model

- **Feed the Data**: Use the training data to train the model. This is typically done in batches.
- **Epochs**: An epoch is one complete pass through the entire training dataset. Choose the number of epochs based on when you observe the validation loss plateauing.
- **Validation Data**: Use a separate validation set (not used in training) to tune the hyperparameters and avoid overfitting.
- **Callbacks**: Implement callbacks for actions like model checkpointing (to save the model at different stages) and early stopping (to stop training when the validation loss stops improving).

### 5. Evaluate the Model

- **Test Set**: After training, evaluate the model's performance on a test set to check how well it generalizes to unseen data.
- **Metrics**: Review the metrics (e.g., accuracy, precision, recall) to assess the model's performance.

### 6. Fine-Tuning and Optimization

- **Hyperparameter Tuning**: Adjust hyperparameters like learning rate, batch size, and architecture specifics to improve performance.
- **Transfer Learning**: Consider using a pre-trained model as the base of your CNN to leverage learned features from large datasets.

### Example Code Snippet

Here's a very simplified example of defining and compiling a CNN model using TensorFlow and Keras:

```python
6f160a83-b859-4f94-962d-0aa6e77203fc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model = Sequential([
    Conv2D(32, (3, 3), input_shape=(64, 64, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(num_classes),
    Activation('softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

```

Remember, the specific architecture, hyperparameters, and training configurations depend heavily on the problem you're solving and the dataset you're working with.

# characteristics of overfitting and underfitting in machine learning models:

| Characteristic | Overfitting | Underfitting |
|----------------|-------------|--------------|
| **Definition** | When a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. | When a model cannot capture the underlying trend of the data. It performs poorly on both training and new data. |
| **Cause** | Too complex model with too many parameters. | Too simple model that doesn’t have enough capacity to learn from the data. |
| **Training Data Performance** | High accuracy | Poor accuracy |
| **Validation/Test Data Performance** | Poor accuracy due to the model being unable to generalize well. | Poor accuracy as the model has not learned the data well. |
| **Signs** | - Very low training error but high validation/test error.<br>- The model captures noise and random fluctuations in the training data. | - High training error and similarly high validation/test error.<br>- The model is too simple to capture the complexity of the data. |
| **Solutions** | - Simplify the model by reducing the number of parameters.<br>- Use regularization techniques (L1, L2, dropout).<br>- Increase training data.<br>- Use data augmentation.<br>- Early stopping during training. | - Increase model complexity.<br>- Add more features to the training data.<br>- Reduce regularization.<br>- Use a more sophisticated model. |

