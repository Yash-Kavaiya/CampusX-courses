# Session 1 - Introduction to GEN AI and LLMs

## Large Language Model
LLM stands for "Large Language Model." It refers to advanced natural language processing (NLP) models that are trained on vast amounts of text data to understand and generate human-like language.

Some key points about Large Language Models (LLMs):
- They use deep learning techniques, especially transformer-based architectures like GPT (Generative Pre-trained Transformer), to process and model language.
- They are trained on massivetext corpora comprising billions or trillions of words from the internet, books, articles, and other sources.
- This large-scale training allows LLMs to develop deep language understanding and generation capabilities.
- Well-known examples of LLMs include GPT-3 by OpenAI, LaMDA by Google, PaLM by Google, BLOOM by Hugging Face, and Claude by Anthropic.
- LLMs can be employed for various NLP tasks like text generation, translation, summarization, question answering, and analysis.
- However, LLMs can also display biases, inconsistencies, and hallucinations present in their training data without proper controls.

## Types of LLM

| Feature | Open-Source LLM | Paid LLM |
|---------|-----------------|----------|
| **Access** | Open-source LLMs are freely available and can be modified and customized by anyone¹[7]²[8]³[10]. | Paid LLMs require a license to use, which often comes with restrictions on usage⁴[1]. |
| **Cost** | There is no cost to use open-source LLMs, making them accessible to a wider audience¹[7]²[8]. | Paid LLMs involve licensing fees, which can be a significant expense, especially for small and medium enterprises⁴[1]. |
| **Customization** | Open-source LLMs offer the flexibility to be deployed on one's own infrastructure and fine-tuned to fit specific needs¹[7]²[8]. | Customization in paid LLMs may be limited by the terms of the license agreement⁴[1]. |
| **Transparency** | Open-source LLMs provide transparency with publicly available training datasets, model architectures, and weights¹[7]²[8]. | Paid LLMs may offer limited information on the mechanisms behind the technology due to proprietary restrictions⁴[1]. |
| **Community Support** | Open-source LLMs benefit from community contributions, which can lead to rapid innovation and improvements¹[7]²[8]. | Paid LLMs are typically supported by the company that owns them, which may limit the scope of community-driven enhancements⁴[1]. |
| **Data Security** | Using open-source LLMs allows companies to maintain full control over their data, enhancing data security and privacy¹[7]. | With paid LLMs, there is a risk of data leaks or unauthorized access to sensitive data by the LLM provider⁴[1]. |

Open-source and paid LLMs each have their own set of advantages and considerations. The choice between them depends on the specific needs, resources, and goals of the user or organization.

## LLM Vocab

1. **Prompt**:
   - In the context of Large Language Models (LLMs), a prompt refers to the initial text or instruction given to the model to guide its generation or response.
   - The prompt serves as a starting point or context for the LLM, and its generation continues based on the prompt.
   - Prompts can be as simple as a single word or phrase, or they can be more complex and structured, providing detailed instructions or context.
   - Effective prompting is crucial for obtaining desired and coherent outputs from LLMs, as the model's response is highly dependent on the prompt.

2. **Context Window**:
   - The context window, also known as the attention window or context length, refers to the maximum number of tokens (words or subwords) that an LLM can consider at once when generating text.
   - LLMs have a limited context window due to computational constraints and memory limitations.
   - For example, GPT-3 has a context window of 2048 tokens, meaning it can only attend to the previous 2048 tokens when generating the next token.
   - If the input text exceeds the context window, the LLM may lose relevant context and generate incoherent or inconsistent text.
   - Increasing the context window size can improve the model's performance on tasks that require long-range dependencies or complex reasoning, but it comes at a higher computational cost.

3. **LLM Hyperparameters**:
   - Hyperparameters are configurations or settings that control the training process and behavior of an LLM.
   - Some common hyperparameters for LLMs include:
     - **Model Size**: The number of parameters (weights) in the model, which determines its capacity and complexity (e.g., GPT-3 has 175 billion parameters).
     - **Batch Size**: The number of training examples processed simultaneously during each iteration of training.
     - **Learning Rate**: The step size at which the model's parameters are updated during training.
     - **Dropout Rate**: The fraction of neurons randomly dropped during training to prevent overfitting.
     - **Attention Heads**: The number of attention heads in the transformer architecture, which controls how the model attends to different parts of the input.
     - **Layer Size**: The number of layers (transformer blocks) in the model, which affects its depth and expressive power.
     - **Tokenizer**: The algorithm used to split the input text into tokens (e.g., subword tokenizers like BPE or WordPiece).
   - Tuning these hyperparameters can significantly impact the model's performance, training time, and resource requirements.
   - Hyperparameter tuning is often done through techniques like grid search, random search, or more advanced methods like Bayesian optimization.

Proper selection and tuning of prompts, context windows, and hyperparameters are crucial for effectively utilizing the capabilities of LLMs and optimizing their performance for specific tasks and scenarios.

## LLAMAINDEX

LLAMA Index is an open-source library developed by Anthropic that helps in building data-aware applications using Large Language Models (LLMs). It provides a way to structure and interact with external data sources, enabling LLMs to generate responses based on both their pre-trained knowledge and the provided data.

Here are some key details about LLAMA Index:

1. **Data Ingestion**: LLAMA Index can ingest various types of data sources, including text files, PDFs, CSV files, and even structured data like Python objects or databases. This allows users to combine information from multiple sources and make it available to the LLM.

2. **Data Structuring**: The ingested data is then structured into an index, which is a data structure optimized for efficient retrieval and querying. LLAMA Index supports different indexing strategies, such as vector-based indexing (using embeddings) or tree-based indexing (using hierarchical structures).

3. **Query Composition**: Users can compose queries by providing a natural language prompt, and LLAMA Index will retrieve relevant information from the indexed data sources. This allows the LLM to generate responses that are informed by both its pre-trained knowledge and the specific data provided.

4. **Response Generation**: LLAMA Index integrates with various LLMs, including GPT-3, LaMDA, PaLM, and others. It passes the composed query and retrieved data to the LLM, which then generates a response based on the provided context.

5. **Customization**: LLAMA Index provides various customization options, such as query preprocessing, result postprocessing, and custom retrieval strategies. This allows users to tailor the behavior of the system to their specific needs.

6. **Iterative Refinement**: LLAMA Index supports iterative refinement, where the user can provide feedback on the generated responses, and the system can use this feedback to refine the query and improve subsequent responses.

7. **Applications**: LLAMA Index can be used for a wide range of applications, including question-answering systems, document summarization, knowledge base construction, and data-aware chatbots or virtual assistants.

By combining the power of LLMs with structured external data sources, LLAMA Index enables the development of more data-aware and context-specific language applications. It abstracts away the complexities of data handling and retrieval, allowing developers to focus on building higher-level applications that leverage both pre-trained language models and relevant external data.

RAG (Retrieval-Augmented Generation) is an approach to enhancing language models with external knowledge retrieval capabilities. It was introduced by researchers at Microsoft and allows language models to generate responses that are informed not only by their pre-trained knowledge but also by relevant information retrieved from external sources.

Here's a detailed explanation of how RAG works:

1. **Architecture**:
   - RAG consists of two main components: a retriever and a generator.
   - The retriever is responsible for retrieving relevant information from external knowledge sources, such as Wikipedia or other document databases.
   - The generator is a pre-trained language model, typically a transformer-based model like GPT-2 or BART, which generates the final output.

2. **Retrieval Process**:
   - When a user provides an input query, the retriever component uses techniques like sparse vector search or dense vector search to retrieve relevant passages or documents from the external knowledge source.
   - The retrieval process can involve different strategies, such as BM25 (a popular text retrieval algorithm) or dense retrieval using learned embeddings.
   - The top-k most relevant passages or documents are selected for further processing.

3. **Generation Process**:
   - The selected passages or documents are concatenated with the original input query to form the context for the generator.
   - The generator (the pre-trained language model) takes this context as input and generates the final output, which can be a natural language response, a summary, or any other desired text output.
   - During generation, the model can attend to both the input query and the retrieved passages, allowing it to incorporate relevant external knowledge into its output.

4. **Training**:
   - The retriever and generator components can be trained jointly or separately, depending on the specific implementation.
   - The retriever can be trained using techniques like negative sampling or contrastive learning to improve its ability to retrieve relevant information.
   - The generator can be fine-tuned on a combination of the pre-training data and examples with retrieved passages, to learn how to effectively integrate external knowledge into its generations.

5. **Applications**:
   - RAG has been successfully applied to tasks like open-domain question answering, where the model needs to retrieve relevant information from external sources to answer complex queries.
   - It can also be used for other tasks that require incorporating external knowledge, such as dialog systems, summarization, or knowledge-grounded generation.

RAG addresses a key limitation of traditional language models, which are often limited by the knowledge contained in their training data. By augmenting the model with external knowledge retrieval capabilities, RAG allows for more informed and knowledge-grounded generations, making language models more versatile and capable of handling a wider range of tasks.