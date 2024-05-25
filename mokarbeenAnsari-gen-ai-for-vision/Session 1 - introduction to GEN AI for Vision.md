# Session 1 - Introduction to GEN AI for Vision

### Course Objective
- Introduction to Generative Al.
- Foundation of Image Representation.
- Diffusion Model Basics.
- Prompt Engineering for Images.
- Project.

### Agenda

- What is AI ?
- Discriminative AI vs Generative AI.
- What is prompt engineering?
- Language model & vision model.
- What is image generation model?
- Unconditional vs Conditional image generation.
- List of vision model.
- Generative AI Workflow.

## What is AI?
Artificial Intelligence (AI) is a broad field that encompasses the theory and development of computer systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

Your explanation is mostly accurate, but let me provide a more comprehensive overview:

1. AI is a branch of computer science that aims to create intelligent machines capable of mimicking human cognitive functions like learning, problem-solving, reasoning, and decision-making.

2. Machine Learning (ML) is a subset of AI that deals with the development of algorithms and statistical models that enable systems to learn from data and improve their performance on a specific task over time, without being explicitly programmed.

3. Deep Learning (DL) is a subfield of Machine Learning that is inspired by the structure and function of the human brain. It involves training artificial neural networks with multiple layers to learn and make intelligent decisions from vast amounts of data.

4. AI models, or AI systems, are computer programs or algorithms designed to perform specific tasks by leveraging techniques from various AI disciplines, such as Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, and others.

It's important to note that while AI aims to mimic human intelligence, it does not necessarily replicate the cognitive processes of the human brain. AI systems operate based on the principles of mathematics, logic, and statistical models, rather than biological mechanisms.

## Discriminative AI vs Generative AI

| Aspect | Discriminative AI | Generative AI |
|--------|-------------------|---------------|
| **Definition** | Discriminative AI models learn to distinguish or classify input data into predefined categories or labels. | Generative AI models learn the underlying patterns and distributions of the input data to generate new data samples that resemble the training data. |
| **Task** | Discrimination tasks, such as classification, regression, and prediction. | Generation tasks, such as text generation, image synthesis, music composition, and data augmentation. |
| **Model Training** | Trained on labeled data, where the model learns to map inputs to corresponding outputs or labels. | Trained on unlabeled data, where the model learns the probability distribution of the data. |
| **Objective** | To find the decision boundary that separates different classes or categories. | To learn the joint probability distribution of the input data and generate new samples from that distribution. |
| **Examples** | Image classification, spam detection, sentiment analysis, and speech recognition. | Text generation (e.g., language models), image generation (e.g., GANs), music composition, and data augmentation. |
| **Evaluation Metrics** | Accuracy, precision, recall, F1-score, and other classification metrics. | Perceptual quality, diversity, and similarity to the training data distribution. |
| **Limitations** | Cannot generate new data samples and may struggle with out-of-distribution or unseen data. | Can generate implausible or unrealistic samples, and training can be computationally expensive. |

**Detailed Explanation:**

1. **Discriminative AI**: These models are trained to classify or discriminate input data into predefined categories or labels. The training process involves feeding the model labeled data, where the inputs are mapped to their corresponding outputs or labels. The objective is to learn the decision boundary that separates different classes or categories. Examples of discriminative models include logistic regression, support vector machines (SVMs), decision trees, and certain types of neural networks used for classification tasks.

2. **Generative AI**: These models learn the underlying patterns and distributions of the input data to generate new data samples that resemble the training data. The training process involves feeding the model unlabeled data, and the model learns the joint probability distribution of the input features. The objective is to capture the characteristics of the training data distribution so that new samples can be generated from that distribution. Examples of generative models include variational autoencoders (VAEs), generative adversarial networks (GANs), and certain types of language models used for text generation or image synthesis.

Discriminative models are better suited for tasks that involve classification, prediction, or decision-making, where the goal is to assign input data to predefined categories or make predictions based on labeled data. Generative models, on the other hand, are more suitable for tasks that involve generating new data samples, such as text generation, image synthesis, or data augmentation.

## What is Prompt Engineering?
Prompt engineering refers to the process of carefully crafting the input prompts or queries that are fed into large language models (LLMs) or other AI systems to elicit desired outputs or behaviors. It is a crucial aspect of getting the most out of these powerful AI models and tailoring their responses to specific tasks or use cases.

The key aspects of prompt engineering include:

1. **Prompt Design**: Crafting the initial prompt or query in a way that provides the necessary context, instructions, and formatting to guide the AI model towards generating the desired type of output. This may involve techniques like few-shot learning, where a few examples of the desired output are included in the prompt.

2. **Prompt Tuning**: Iteratively refining and adjusting the prompt based on the model's outputs to improve the relevance, accuracy, and quality of the generated responses. This may involve adding or removing examples, rephrasing instructions, or incorporating additional context.

3. **Prompt Combination**: Combining multiple prompts or prompt components to handle complex tasks or capture different aspects of the desired output. This can involve techniques like chain-of-thought prompting, where the model is guided through a step-by-step reasoning process.

4. **Prompt Filtering**: Incorporating techniques to filter out or mitigate undesirable outputs, such as biases, hallucinations, or unsafe content, by including specific instructions or constraints in the prompt.

5. **Prompt Adaptation**: Adapting or fine-tuning the prompt engineering techniques for specific domains, tasks, or use cases to optimize the model's performance and outputs for those scenarios.

Effective prompt engineering can significantly improve the quality, consistency, and reliability of the outputs generated by large language models and other AI systems. It is particularly important for applications where accurate and tailored responses are critical, such as in customer service, content generation, or decision support systems.

| Aspect | Language Model | Vision Model |
|--------|-----------------|--------------|
| **Input Data** | Text data (sequences of words, characters, or tokens) | Visual data (images, videos) |
| **Task** | Natural language processing tasks, such as text generation, translation, summarization, and question answering | Computer vision tasks, such as image classification, object detection, semantic segmentation, and image generation |
| **Model Architecture** | Transformers (e.g., BERT, GPT), Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs) | Convolutional Neural Networks (CNNs), Region-based Convolutional Neural Networks (R-CNNs), Generative Adversarial Networks (GANs) |
| **Training Data** | Large text corpora (e.g., web pages, books, articles) | Labeled image datasets (e.g., ImageNet, COCO, Pascal VOC) |
| **Preprocessing** | Tokenization, text cleaning, embeddings | Image resizing, normalization, data augmentation |
| **Evaluation Metrics** | Perplexity, BLEU, ROUGE, F1-score | Accuracy, mAP, IoU, FID score |
| **Applications** | Language translation, chatbots, text summarization, content generation | Image recognition, object detection, autonomous vehicles, medical imaging |
| **Challenges** | Handling ambiguity, context understanding, common sense reasoning | Occlusion, viewpoint variation, fine-grained recognition |
| **Ethical Considerations** | Biases, hallucinations, potential misuse (e.g., fake news, hate speech) | Privacy concerns, algorithmic bias, environmental impact |

**Detailed Explanation:**

1. **Language Model**: Language models are trained on text data to learn patterns and relationships in natural language. They are designed to understand and generate human-readable text. Common architectures include Transformers (e.g., BERT, GPT), Recurrent Neural Networks (RNNs), and Long Short-Term Memory (LSTMs). These models are trained on large text corpora, such as web pages, books, and articles, and are evaluated using metrics like perplexity, BLEU, ROUGE, and F1-score. Applications include language translation, chatbots, text summarization, and content generation. However, language models can struggle with ambiguity, context understanding, and common sense reasoning. They also raise ethical concerns regarding biases, hallucinations, and potential misuse (e.g., fake news, hate speech).

2. **Vision Model**: Vision models are trained on visual data, such as images and videos, to perform computer vision tasks like image classification, object detection, semantic segmentation, and image generation. Common architectures include Convolutional Neural Networks (CNNs), Region-based Convolutional Neural Networks (R-CNNs), and Generative Adversarial Networks (GANs). These models are trained on labeled image datasets, such as ImageNet, COCO, and Pascal VOC, and are evaluated using metrics like accuracy, mean Average Precision (mAP), Intersection over Union (IoU), and Fréchet Inception Distance (FID) score. Applications include image recognition, object detection, autonomous vehicles, and medical imaging. Vision models can struggle with occlusion, viewpoint variation, and fine-grained recognition. Ethical considerations include privacy concerns, algorithmic bias, and environmental impact due to the computational resources required for training and inference.

While language models and vision models are designed for different modalities (text and visual data, respectively), there is growing interest in multimodal models that can handle and combine multiple data types, such as text and images. Additionally, techniques like transfer learning and self-supervised learning are being explored to leverage knowledge from one domain to improve performance in another.

## What is image generation model?

An image generation model is a type of artificial intelligence model that is designed to generate new images from a given input, such as text descriptions, random noise, or other images. These models are trained on large datasets of images and learn to understand the underlying patterns and distributions within the data.

The input to an image generation model can take various forms:

1. **Text-to-Image Generation**: The model takes a text description as input and generates an image that corresponds to the given textual description. For example, generating an image of a "sunset over a beach with palm trees" based on the text input.

2. **Noise-to-Image Generation**: The model takes random noise (e.g., a vector of random numbers) as input and generates a coherent image from that noise. This is often used in generative adversarial networks (GANs) and variational autoencoders (VAEs).

3. **Image-to-Image Generation**: The model takes an existing image as input and generates a new image based on that input. This can be used for tasks like image translation (e.g., converting a sketch to a photorealistic image), image inpainting (filling in missing or corrupted parts of an image), or image editing (changing certain aspects of an image).

Some popular architectures used for image generation include:

1. **Generative Adversarial Networks (GANs)**: These models consist of two neural networks, a generator and a discriminator, that are trained in an adversarial manner. The generator learns to produce realistic images, while the discriminator learns to distinguish between real and generated images.

2. **Variational Autoencoders (VAEs)**: These models learn to encode input data (e.g., images) into a latent space and then decode from that latent space to generate new images.

3. **Diffusion Models**: These models learn to generate images by iteratively denoising random noise, gradually introducing more structure and detail to the generated image.

Image generation models have applications in various domains, such as creative industries (generating art, design, or advertising materials), gaming (generating synthetic environments or characters), and research (generating synthetic data for training other models).

It's important to note that while image generation models have made significant advancements, they can still produce unrealistic or biased outputs, and there are ongoing efforts to address these limitations and ensure responsible and ethical use of these technologies.

## Unconditional Image Generation

Unconditional image generation refers to the process of generating new images from random noise or a latent space representation, without any specific conditioning input like text or an existing image. These models learn the underlying distribution of the training data and can produce diverse and realistic images without requiring additional guidance or constraints.

Here's an explanation of unconditional image generation, with examples of popular models:

1. **Generative Adversarial Networks (GANs)**: GANs are a popular approach for unconditional image generation. They consist of two neural networks: a generator and a discriminator. The generator takes random noise as input and generates fake images, while the discriminator tries to distinguish between real images from the training data and the fake images generated by the generator. Through this adversarial training process, the generator learns to produce increasingly realistic images that can fool the discriminator.

2. **Denoising Diffusion Probabilistic Models (DDPMs)**: DDPMs, such as the model used by [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/), are a class of generative models that learn to gradually denoise random noise to generate high-quality images. The training process involves gradually adding noise to real images and training the model to reverse this process, effectively learning to generate new images from pure noise.

Both GANs and DDPMs have demonstrated impressive results in unconditional image generation, producing realistic and diverse images of faces, landscapes, objects, and more, without any conditioning input.

Unconditional image generation models are often used in creative applications, such as generating synthetic data for training other models, creating artwork or design elements, and exploring the capabilities of generative models. However, these models can also raise ethical concerns, such as the potential for generating deepfakes or biased representations, which highlights the need for responsible development and deployment of these technologies.

It's worth noting that while unconditional image generation models can produce impressive results, they may struggle to generate images with specific attributes or content without additional conditioning or guidance. For such tasks, conditional image generation models, which take additional inputs like text descriptions or reference images, are often more suitable.

## Conditional Image Generation
Conditional image generation refers to the process of generating new images based on some form of conditioning input, such as text descriptions, class labels, or reference images. These models take the conditioning input into account and generate images that conform to the given constraints or specifications.

Here's an explanation of conditional image generation, with an example of a popular model:

1. **Text-to-Image Generation**: One of the most common forms of conditional image generation is text-to-image generation, where the model takes a textual description as input and generates an image that corresponds to that description. For example, "Generate a dog image wearing a hat and riding a cycle."

2. **Class-Conditional Generation**: In this case, the model takes a class label or category as input and generates an image belonging to that class or category. For instance, generating images of various dog breeds by providing the breed name as the conditioning input.

3. **Image-to-Image Translation**: Here, the conditioning input is an existing image, and the model generates a new image based on that input. This can be used for tasks like style transfer (e.g., converting a photo to a painting-like style), image inpainting (filling in missing or corrupted parts of an image), or image editing (changing specific aspects of an image).

A popular example of a conditional image generation model is Stable Diffusion, developed by Stability AI and available on the Hugging Face platform (https://huggingface.co/spaces/stabilityai/stable-diffusion). Stable Diffusion is a latent diffusion model that can generate high-quality images based on textual descriptions or conditioning inputs.

To use Stable Diffusion for conditional image generation, you can provide a textual prompt describing the desired image, such as "Generate a dog image wearing a hat and riding a cycle." The model will then generate an image that attempts to match the given description, combining its understanding of the textual input with its learned knowledge of image composition and visual concepts.

Conditional image generation models have numerous applications, including creative content generation, data augmentation for computer vision tasks, and visualizing concepts or ideas described in text. However, like other generative models, they can also raise ethical concerns regarding biases, intellectual property rights, and potential misuse.

It's important to note that while conditional image generation models have made significant progress, they may still produce imperfect or biased outputs, especially for complex or abstract prompts. Ongoing research aims to improve the quality, consistency, and controllability of these models, as well as address potential ethical implications.

## Popular Vision Model

Most popular and influential vision models for image generation tasks, particularly conditional image generation based on text prompts. Here's a brief overview of each:

1. **Midjourney by Midjourney**:
   - A text-to-image model that generates highly detailed and creative images from textual descriptions.
   - Widely used by artists, designers, and creatives for generating unique visual content.
   - Employs a proprietary AI architecture developed by the Midjourney team.

2. **DALL-E by OpenAI**:
   - One of the pioneering text-to-image models, capable of generating diverse and realistic images from textual prompts.
   - Utilizes a transformer-based architecture trained on a vast dataset of text-image pairs.
   - Introduced by OpenAI and has been influential in advancing the field of multimodal AI.

3. **Stable Diffusion by Stability AI**:
   - An open-source text-to-image model based on latent diffusion models.
   - Offers high-quality image generation capabilities while being more computationally efficient than some other models.
   - Widely adopted by researchers, developers, and creators due to its open-source nature and strong performance.

4. **Imagen by Google**:
   - A text-to-image model developed by Google Brain, known for its ability to generate high-resolution and highly detailed images.
   - Employs a specialized architecture called the Diffusion Model with Denoising Strength Modulation.
   - Capable of generating images with exceptional fidelity and coherence, even for complex prompts.

These models have democratized the creation of visual content by allowing users to generate unique and imaginative images from natural language descriptions. They have found applications in various domains, including art, design, advertising, education, and research.

## Generative AI Workflow
Here's a comparison of the traditional Machine Learning/Deep Learning (ML/DL) workflow and the Generative AI (GenAI) workflow, presented in a markdown table:

| Aspect | ML/DL Workflow | GenAI Workflow |
|--------|-----------------|-----------------|
| **Goal** | Learn patterns from data to make predictions or classifications | Generate new data samples that resemble the training data distribution |
| **Data** | Labeled or unlabeled data, often structured and curated | Large, diverse, and potentially unstructured data (e.g., text, images, audio) |
| **Model Training** | Supervised learning (labeled data), unsupervised learning (unlabeled data), or a combination | Self-supervised learning, contrastive learning, or adversarial training |
| **Model Architecture** | Task-specific architectures (e.g., CNNs for images, RNNs for sequences) | Large, general-purpose models (e.g., Transformers, Diffusion Models) |
| **Evaluation** | Task-specific metrics (e.g., accuracy, F1-score, IoU) | Perceptual quality, diversity, and fidelity to the training data distribution |
| **Deployment** | Inference on new data for prediction or classification tasks | Sampling or generation of new data samples (e.g., images, text, audio) |
| **Iterative Process** | Model selection, hyperparameter tuning, and error analysis | Prompt engineering, prompt tuning, and output filtering/curation |
| **Challenges** | Dealing with limited or biased data, overfitting, and generalization | Controlling for biases, hallucinations, and undesirable outputs |
| **Applications** | Image classification, object detection, machine translation, recommender systems | Text generation, image synthesis, music composition, data augmentation |

**Detailed Explanation:**

1. **Machine Learning/Deep Learning Workflow**:
   - The goal is to learn patterns from data to make accurate predictions or classifications on new, unseen data.
   - The data is often labeled or structured for supervised learning tasks, or unlabeled for unsupervised learning.
   - Model training involves learning from the provided data, using techniques like supervised learning, unsupervised learning, or a combination.
   - Model architectures are typically task-specific, such as Convolutional Neural Networks (CNNs) for image tasks or Recurrent Neural Networks (RNNs) for sequence tasks.
   - Evaluation is based on task-specific metrics like accuracy, F1-score, or Intersection over Union (IoU).
   - Once trained, the model is deployed for inference on new data to make predictions or classifications.
   - The process is iterative, involving model selection, hyperparameter tuning, and error analysis to improve performance.
   - Challenges include dealing with limited or biased data, overfitting, and ensuring good generalization to new, unseen data.

2. **Generative AI Workflow**:
   - The goal is to generate new data samples that resemble the training data distribution, such as generating realistic images, text, or audio.
   - The training data is often large, diverse, and potentially unstructured, like text corpora or image datasets.
   - Model training involves self-supervised learning, contrastive learning, or adversarial training techniques, like in Generative Adversarial Networks (GANs) or Diffusion Models.
   - Model architectures are typically large, general-purpose models like Transformers or Diffusion Models, capable of handling diverse data types.
   - Evaluation focuses on the perceptual quality, diversity, and fidelity of the generated samples to the training data distribution.
   - Once trained, the model is used to sample or generate new data samples, such as images, text, or audio.
   - The process involves prompt engineering, prompt tuning, and output filtering/curation to control the generation process and improve output quality.
   - Challenges include controlling for biases, hallucinations, and undesirable outputs, as well as addressing potential ethical concerns.
   - Applications include text generation, image synthesis, music composition, and data augmentation.

While the traditional ML/DL workflow focuses on learning patterns for prediction or classification tasks, the GenAI workflow emphasizes generating new, realistic data samples that resemble the training data distribution. Both workflows involve iterative processes but with different goals, techniques, and challenges.