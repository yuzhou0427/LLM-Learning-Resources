# Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face


地址：https://huggingface.co/blog/mixtral

github：https://github.com/huggingface/blog/blob/main/mixtral.md


<br>

&nbsp;&nbsp;&nbsp;&nbsp;Mixtral 8x7b是Mistral今天发布的一款令人兴奋的大型语言模型，它在开放获取模型方面取得了新的技术水平，并在许多基准测试中胜过了GPT-3.5。我们很高兴能够通过在Hugging Face生态系统中全面集成Mixtral来支持这一发布。


<br>

## What is Mixtral 8x7b?

&nbsp;&nbsp;&nbsp;&nbsp;Mixtral与Mistral 7B拥有相似的架构，但有一个独特之处：它实际上是8个“专家”模型的组合，这得益于一种称为“专家混合”（Mixture of Experts，MoE）的技术。对于Transformer模型，这种技术的工作原理是通过用稀疏的MoE层替换一些前馈层。MoE层包含一个路由网络，选择哪些专家最有效地处理哪些标记。在Mixtral的情况下，为每个时间步选择了两个专家，这使得该模型能够以12B参数密集型模型的速度解码，尽管其包含的有效参数数量是其4倍！

> 有关MoE的更多详细信息，请参阅我们的相关博客文章：https://huggingface.co/blog/moe


**Mixtral发布简要摘要：**

- 发布基础版和Instruct版本
- 支持32,000标记的上下文长度。
- 在大多数基准测试中胜过Llama 2 70B，与GPT3.5相匹敌或超越
- 支持英语、法语、德语、西班牙语和意大利语。
- 在编码方面表现良好，在HumanEval上达到40.2%
- 具有商业宽松的Apache 2.0许可证


&nbsp;&nbsp;&nbsp;&nbsp;Mixtral模型有多好呢？以下是base model在[LLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)上与其他开放模型相比的性能概述（分数越高越好）：

![Mixtral-8x7b-basemodel-rank.png](..%2Fassets%2FMixtral-8x7b-basemodel-rank.png)


<br>

&nbsp;&nbsp;&nbsp;&nbsp;对于Instruct和Chat模型，评估MT-Bench或AlpacaEval等基准更为合适。下面，我们展示了[Mixtral Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)在与顶级闭源和开源模型的性能比较中的表现（分数越高越好）：

| Model                                                                                               | Availability    | Context window (tokens) | MT-Bench score ⬇️ |
| --------------------------------------------------------------------------------------------------- | --------------- | ----------------------- | ---------------- |
| [GPT-4 Turbo](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)        | Proprietary     | 128k                    | 9.32             |
| [GPT-3.5-turbo-0613](https://platform.openai.com/docs/models/gpt-3-5)                               | Proprietary     | 16k                     | 8.32             |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | Apache 2.0      | 32k                     | 8.30             |
| [Claude 2.1](https://www.anthropic.com/index/claude-2-1)                                            | Proprietary     | 200k                    | 8.18             |
| [openchat/openchat_3.5](https://huggingface.co/openchat/openchat_3.5)                               | Apache 2.0      | 8k                      | 7.81             |
| [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)                 | MIT             | 8k                      | 7.34             |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)             | Llama 2 license | 4k                      | 6.86             |


令人印象深刻的是，Mixtral Instruct在MT-Bench上胜过所有其他开源模型，并成为第一个与GPT-3.5性能相媲美的模型！


### About the name

&nbsp;&nbsp;&nbsp;&nbsp;Mixtral MoE被称为 Mixtral-8x7B，但它并没有56B的参数。在发布后不久，我们发现一些人被误导以为这个模型的行为类似于包含8个每个有7B参数的模型的集成模型，但MoE模型的工作方式并非如此。模型的只有一些层（前馈块）被复制；其余的参数与7B模型相同。总参数数不是56B，而是约45B。一个更好的名称可能是 [Mixtral-45-8e](https://twitter.com/osanseviero/status/1734248798749159874)，以更好地传达架构。有关MoE如何工作的更多详细信息，请参阅我们的“[深入解析专家混合](https://huggingface.co/blog/moe)”文章。



### Prompt format

&nbsp;&nbsp;&nbsp;&nbsp;The [base model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) has no prompt format. Like other base models, it can be used to continue an input sequence with a plausible continuation or for zero-shot/few-shot inference. It’s also a great foundation for fine-tuning your own use case. The [Instruct model](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) has a very simple conversation structure.

```bash
<s> [INST] User Instruction 1 [/INST] Model answer 1</s> [INST] User instruction 2[/INST]
```

该格式必须完全复制以实现有效使用。稍后我们将展示如何使用 `transformers` 中提供的聊天模板轻松复制 instruct 提示。


## Demo

&nbsp;&nbsp;&nbsp;&nbsp;您可以在 Hugging Face Chat 与 Mixtral Instruct 模型聊天！在这里查看: https://huggingface.co/chat/?model=mistralai/Mixtral-8x7B-Instruct-v0.1.


## Inference

我们提供了两种主要方式来运行 Mixtral 模型的推理：

- 通过 🤗 Transformers 的 pipeline() 函数。
- 使用文本生成推理（Text Generation Inference），支持连续批处理、张量并行等高级功能，实现极快的推理结果。


对于每种方法，可以在半精度（float16）或使用量化权重的情况下运行模型。由于 Mixtral 模型在大小上大致相当于一个参数为 45B 的稠密模型，我们可以按照以下方式估算所需的最小 VRAM：

| Precision | Required VRAM |
| --------- | ------------- |
| float16   | >90 GB        |
| 8-bit     | >45 GB        |
| 4-bit     | >23 GB        |


### Using 🤗 Transformers

使用 transformers [release 4.36](https://github.com/huggingface/transformers/releases/tag/v4.36.0)，您可以使用 Mixtral 并充分利用 Hugging Face 生态系统中的所有工具，例如：

- 训练和推断脚本以及示例
- 安全文件格式 (`safetensors`)
- 与诸如 bitsandbytes（4 位量化）、PEFT（参数高效微调）和 Flash Attention 2 等工具的集成
- 运行模型生成的实用程序和辅助工具
- 导出模型以进行部署的机制

确保使用最新的 `transformers` 版本：

```bash
pip install -U "transformers==4.36.0" --upgrade
```

在以下代码片段中，我们展示了如何使用 🤗 Transformers 进行推断并进行 4 位量化。由于模型体积较大，您需要至少具有 30 GB RAM 的显卡来运行它。这包括 A100（80 或 40GB 版本）或 A6000（48 GB）。

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

> \<s>[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST] A
Mixture of Experts is an ensemble learning method that combines multiple models,
or "experts," to make more accurate predictions. Each expert specializes in a
different subset of the data, and a gating network determines the appropriate
expert to use for a given input. This approach allows the model to adapt to
complex, non-linear relationships in the data and improve overall performance.
> 

### Using Text Generation Inference

**[Text Generation Inference](https://github.com/huggingface/text-generation-inference)** 是 Hugging Face 开发的一个可用于生产的推理容器，旨在实现大型语言模型的轻松部署。它具有连续批处理、令牌流式传输、在多个 GPU 上进行快速推理的张量并行等特性，同时提供生产就绪的日志记录和追踪。

您可以在 Hugging Face [Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMixtral-8x7B-Instruct-v0.1&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000) 上部署 Mixtral , 其使用 Text Generation Inference 作为后端。要部署 Mixtral 模型，请转到 [model page](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 然后点击 [Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=meta-llama/Llama-2-7b-hf) .

*Note: 您可能需要通过电子邮件请求配额升级，以便访问 A100 GPU，邮箱地址为 **[api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co)**。*


您可以在我们的博客中了解有关[使用 Hugging Face 推理端点部署 LLMs](https://huggingface.co/blog/inference-endpoints-llm)的更多信息。 **[blog](https://huggingface.co/blog/inference-endpoints-llm)** 包含有关支持的超参数以及如何使用 Python 和 Javascript 流式传输响应的信息。

您还可以通过以下方式在本地使用 Docker 在 2x A100s（80GB）上运行文本生成推理：

```bash
docker run --gpus all --shm-size 1g -p 3000:80 -v /data:/data ghcr.io/huggingface/text-generation-inference:1.3.0 \
	--model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
	--num-shard 2 \
	--max-batch-total-tokens 1024000 \
	--max-total-tokens 32000
```


## Fine-tuning with 🤗 TRL

训练大型语言模型在技术和计算上都具有挑战性。在这一部分，我们将看一下 Hugging Face 生态系统中提供的工具，以在单个 A100 GPU 上高效训练 Mixtral。

以下是在 OpenAssistant 的 [chat dataset](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) 上对 Mixtral 进行微调的示例命令。为了节省内存，我们使用 4-bit 量化和 [QLoRA](https://arxiv.org/abs/2305.14314) 来定位注意力块中的所有线性层。请注意，与密集的 transformer 不同，我们不应该定位 MLP 层，因为它们是稀疏的，与 PEFT 不太互动。

First, install the nightly version of 🤗 TRL and clone the repo to access the [training script](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py):

```bash
pip install -U transformers
pip install git+https://github.com/huggingface/trl
git clone https://github.com/huggingface/trl
cd trl
```

Then you can run the script:

```bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=1 \
	examples/scripts/sft.py \
	--model_name mistralai/Mixtral-8x7B-v0.1 \
	--dataset_name trl-lib/ultrachat_200k_chatml \
	--batch_size 2 \
	--gradient_accumulation_steps 1 \
	--learning_rate 2e-4 \
	--save_steps 200_000 \
	--use_peft \
	--peft_lora_r 16 --peft_lora_alpha 32 \
	--target_modules q_proj k_proj v_proj o_proj \
	--load_in_4bit
```

这需要大约 48 小时在单个 A100 上进行训练，但可以通过调整 `--num_processes` 到你可用的 GPU 数量来轻松并行化。


