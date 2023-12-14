# Multi-modal RAG on slide decks
> 多模态RAG检索增强生成2种实现方式

blog:https://blog.langchain.dev/multi-modal-rag-template/


## Motivation：

检索增强生成（RAG）是LLM应用开发中最重要的概念之一。许多类型的文档可以传递到LLM的上下文窗口中，实现交互式聊天或问答助手。尽管迄今为止的许多RAG应用主要关注文本，但许多信息以视觉内容的形式传达。以幻灯片为例，它们在投资者演示到公司内部沟通等各种用例中很常见。随着多模态LLM（如GPT-4V）的出现，现在可以解锁常常在幻灯片中捕获的视觉内容的RAG。在下面，我们展示了解决这个问题的两种不同方式。我们分享了一个用于评估幻灯片上的RAG的公共基准，并使用它突显这些方法之间的权衡。最后，我们提供了一个快速创建用于幻灯片的多模态RAG应用的模板。


## Design：

该任务类似于文本文档上的RAG应用程序：根据用户的问题检索相关的幻灯片，并将它们传递给多模式LLM（GPT-4V）进行答案合成。有至少两种一般的方法来解决这个问题。

- **`Multi-modal embeddings`：**
    提取幻灯片作为图像，使用多模型嵌入将每个图像嵌入，根据用户输入检索相关的幻灯片图像，并将这些图像传递给GPT-4V进行答案合成。我们先前发布了使用Chroma和OpenCLIP嵌入的这方面的技术指南。
    > 需要使用到图片多模态模型CLIP等。

- **`Multi-vector retriever`：**
    提取幻灯片作为图像，使用GPT-4V对每个图像进行摘要，将图像摘要与原始图像链接嵌入，基于图像摘要与用户输入之间的相似性检索相关图像，最后将这些图像传递给GPT-4V进行答案合成。我们先前发布了使用多向量检索器的这方面的技术指南。



![Multi-modal RAG on slide decks.png](..%2Fassets%2FMulti-modal%20RAG%20on%20slide%20decks.png)


这两种方法之间的权衡很明显：Multi-modal embeddings是一种更简单的设计，反映了我们在基于文本的RAG应用中所做的工作，但选项有限，并且关于检索在视觉上相似的图表或表格的能力存在一些疑问。相比之下，图像摘要使用成熟的文本嵌入模型，并且可以以相当详细的方式描述图表或图形。但是，这种设计面临更高的图像摘要复杂性和成本。



## Evaluation：

我们基准测试中的<问题-答案>对基于幻灯片的视觉内容。我们使用LangSmith评估了上述两种RAG方法，并与使用文本提取的RAG进行了比较（仅文本的Top K RAG）。

|Approach| Score ([CoT accuracy](https://docs.smith.langchain.com/evaluation/evaluator-implementations?ref=blog.langchain.dev#correctness-qa-evaluation)) |
|-|------------------------------------------------------------------------------------------------------------------------------------------------|
|Top k RAG (text only)| 20%                                                                                                                                            |
|Multi-modal embeddings| 60%                                                                                                                                            |
|Multi-vector retriever w/ image summary| 90%                                                                                                                                            |


## Insights:

1. **Multi-modal approaches far exceed the performance of text-only RAG：**
    
    使用多模态方法（60%和90%）相比仅加载文本的RAG（20%），我们看到了明显的改善。

2. **GPT-4V is powerful for structured data extraction from images：**

    GPT-4V能够从幻灯片中正确提取这些信息。

3. **Retrieval of the correct image is the central challenge：**

    如果正确的图像被检索，则GPT-4V通常能够正确回答问题。然而，图像检索是中心挑战。我们发现图像摘要确实在检索方面比多模态嵌入有显著提高，但伴随着预先计算摘要的更高复杂性和成本。中心需求是能够区分视觉上相似幻灯片的多模态嵌入。OpenCLIP有各种不同的模型，值得尝试；它们可以根据此处所示进行轻松配置。








