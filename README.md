<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/img/SambaNova-light-logo-1.png" height="60">
  <img alt="Text changing depending on mode. Light: 'So light!' Dark: 'So dark!'" src="/img/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# BLOOMChat Training Repo

## Overview
This repo contains the data preparation, tokenization, training and inference code for [BLOOMChat-176B-v1](https://huggingface.co/sambanovasystems/BLOOMChat-176B-v1). BLOOMChat is a 176 billion parameter multilingual chat model. It is instruction tuned from [BLOOM (176B)](https://huggingface.co/bigscience/bloom) on assistant-style conversation datasets and supports conversation, question answering and generative answers in multiple languages.



We trained BLOOMChat on [SambaNova DataScale systems](https://sambanova.ai/products/datascale/) using SambaNova’s unique [Reconfigurable Dataflow Architecture](https://sambanova.ai/wp-content/uploads/2021/04/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English.pdf) The training data used to train BLOOMChat originated from [OIG dataset from OpenChatKit](https://huggingface.co/datasets/laion/OIG), [Dolly 2.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k), and [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1).


## Additional Information

- **Blog Post**: [More Information Needed]
- **Discord**: [Link](https://discord.com/invite/8z2Pe7cpRv)
- **HF Hosting**: [More Information Needed]
- **BLOOMCHAT-176B-v1**: [Link](https://huggingface.co/sambanovasystems/BLOOMChat-176B-v1)

## Training Procedure

### Environment setup

- Clone [SambaNova's Generation Data Preparation](https://github.com/sambanova/generative_data_prep) repo
- Create a virtual environment
- Set up environment using the above repo's instructions
- Run this command `pip install datasets`

### Data Preprocessing

Further preprocessing had been done on the original datasets. You can find the relevant code under [data prep](data_prep).

To run these files:
1. `cd data_prep`
2. `python prepare_oa_dolly.py`
3. `python subsample_openchatkit.py`

After running these commands there should be 2 files under the `data_prep` directory:
- `oasst1_dolly.jsonl`
- `bloom_ock_100K_each.jsonl`

NOTE these files are referenced in the tokenization code, so when running the tokenization scripts they need to be done within `tokenization_prep` otherwise the file paths will need to be changed.

### Dataset Tokenization

The next step after preprocessing the data is to tokenize the data using [SambaNova's Generation Data Preparation](https://github.com/sambanova/generative_data_prep) repo. The scripts that utilize this public repository can be found under [tokenization prep](tokenization_prep).

Follow the instructions on how to set up Generation Data Preparation.

For the bash scripts they take in one argument which is the absolute path to the generation data preparation repo.

Example of running the script:

1. `cd tokenization_prep`
2. `bash tokenization_oa_dolly.sh /home/.../generative_data_prep`
3. `bash tokenization_openchatkit.sh /home/.../generative_data_prep`

After running these scripts there should be 2 directories under the `tokenization_prep` directory:
 - `oasst1_dolly_out`
 - `bloom_ock_100k_out`

 These directories contain two directories:

 - `hdf5`
 - `splits`

 The `splits` directory contains the original text, the `hdf5` directory contains the tokenized text which will be fed into the model.

### Training

As our models were trained on SambaNova's in-house Reconfigurable Dataflow Unit (RDU), our scripts will not work for training on GPU; however, we want to give better insight and transparency on the hyper-parameters that were used to train this model. The scripts that were ran on the RDU can be found under [training](training). The [model](https://huggingface.co/bigscience/bloom) is ran directly from HuggingFace, using a built-in wrapper provided by SambaFlow, our SDK for the RDU. For those interested in running models on RDUs, [please feel free to get in touch](https://sambanova.ai/getstarted).

NOTE: BLOOMChat is a two step process:

1. First train a model using [OIG dataset from OpenChatKit](https://huggingface.co/datasets/laion/OIG)
2. Second train the above model using [Dolly 2.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Final Model)

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- [OIG dataset from OpenChatKit](https://huggingface.co/datasets/laion/OIG)
- [Dolly 2.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)

## Quick Start Inference on GPU

[This tutorial](https://github.com/huggingface/transformers-bloom-inference) from Huggingface will be the base layer for running our model. The tutorial is intended for BLOOM; however, since our model is based off of BLOOM we can repurpose it.

For setup instructions follow the Huggingface tutorial.

NOTE: Things that we had to modify in order for BLOOMChat to work:
- Install transformers version 4.27.0
    - `pip install transformers==4.27.0`
- Change the model name from `bigscience/bloom` to `sambanovasystems/BLOOMChat-176B-v1`
- Modifying `inference_server/models/hf_accelerate.py`
    - This is because for our testing of this repo we used 4 80GB A100 GPUs and would run into memory issues
- Modifying `inference_server/cli.py`
    - This is because the model was trained using specific human, bot tags
    - Trailing spaces may lead to subpar performance

Modifications for `inference_server/models/hf_accelerate.py`:

```diff
diff --git a/inference_server/models/hf_accelerate.py b/inference_server/models/hf_accelerate.py
index 9be3c3f..a8ecb1d 100644
--- a/inference_server/models/hf_accelerate.py
+++ b/inference_server/models/hf_accelerate.py
@@ -1,4 +1,5 @@
 from argparse import Namespace
+from accelerate.utils.modeling import get_max_memory
 
 import torch
 
@@ -12,6 +13,12 @@ class HFAccelerateModel(Model):
 
         kwargs = {"pretrained_model_name_or_path": args.model_name, "device_map": "auto"}
 
+        original_max_memory_dict = get_max_memory()
+
+        reduce_max_memory_dict = {device_key: int(original_max_memory_dict[device_key] * 0.85) for device_key in original_max_memory_dict}
+
+        kwargs["max_memory"] = reduce_max_memory_dict
+
         if get_world_size() > 1:
             kwargs["device_map"] = "balanced_low_0"

```

Modifications for `inference_server/cli.py`:

```diff
diff --git a/inference_server/cli.py b/inference_server/cli.py
index fc903d5..5450236 100644
--- a/inference_server/cli.py
+++ b/inference_server/cli.py
@@ -22,6 +22,9 @@ def main() -> None:
     while True:
         input_text = input("Input text: ")
 
+        input_text = input_text.strip()
+        modified_input_text = f"<human>: {input_text}\n<bot>:"
+
         if input("change generate_kwargs? [y/n] ") == "y":
             while True:
                 try:
@@ -33,7 +36,7 @@ def main() -> None:
                     print("message =", e_message)
                     continue
 
-        response = model.generate(text=[input_text], generate_kwargs=generate_kwargs)
+        response = model.generate(text=[modified_input_text], generate_kwargs=generate_kwargs)
 
         print_rank_0("Output text:", response.text[0])
         print_rank_0("Generated tokens:", response.num_generated_tokens[0])

```

Running command for bf16
```
python -m inference_server.cli --model_name sambanovasystems/BLOOMChat-176B-v1 --model_class AutoModelForCausalLM --dtype bf16 --deployment_framework hf_accelerate --generate_kwargs '{"do_sample": false, "temperature": 0.8, "repetition_penalty": 1.2, "top_p": 0.9, "max_new_tokens": 512}'
```
Running command for int8 (sub optimal performance, but fast inference time):
```
python -m inference_server.cli --model_name sambanovasystems/BLOOMChat-176B-v1 --model_class AutoModelForCausalLM --dtype int8 --deployment_framework hf_accelerate --generate_kwargs '{"do_sample": false, "temperature": 0.8, "repetition_penalty": 1.2, "top_p": 0.9, "max_new_tokens": 512}'
```
**DISCLAIMER:** When using int8, the results will be subpar compared to bf16 as the model is being [quantized](https://huggingface.co/blog/hf-bitsandbytes-integration#introduction-to-model-quantization).

### Suggested Inference Parameters
- Temperature: 0.8
- Repetition penalty: 1.2
- Top-p: 0.9
- Max generated tokens: 512

### Suggested Prompts To Try in GPU Tutorial
```
Input text: Write a script in which Bob accidentally breaks his dad's guitar
```

```
Input text: Create an itemized list of tasks to complete to start a clothing brand
```

```
Input text: 十七岁的风是什么颜色的?
```

## Quick Start Inference on RDU

The inference code to run the model can be found under [RDU quick start](rdu_quick_start). This code requires the [SambaFlow](https://docs.sambanova.ai/developer/latest/sambaflow-intro.html) SDK to execute. For those interested in running models on RDUs, [please feel free to get in touch](https://sambanova.ai/getstarted).

## Acknowledgement

We would like to extend our gratitude to [Together](https://www.together.xyz/) for their insightful technical discussions on overall project planning, data processing, model training, human evaluation experiment design, open-source endeavors, and their contributions on data processing code on [OpenChatKit](https://github.com/togethercomputer/OpenChatKit), [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1), and [Dolly 2.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

We would also like to extend our gratitude to Jue Wang who was the main contributor from Together on this collaboration.

We are grateful to the various researchers and open-source projects that have contributed to the development of BLOOMChat. We thank [BigScience](https://bigscience.huggingface.co/) for providing the [BLOOM](https://huggingface.co/bigscience/bloom) model, which served as the base for our instruction tuning. We also thank [LAION](https://laion.ai/) for their [OIG dataset](https://huggingface.co/datasets/laion/OIG), OpenAssistant Conversations Dataset ([OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)) and also thank [Databricks](https://www.databricks.com/) for providing [Dolly 2.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k), to provide the dataset that we instruction tuned on.

## Cite BLOOMChat
```
@software{bloomchat,
  title = {{BLOOMChat: a New Open Multilingual Chat LLM}},
  author = {SambaNova Systems, Together Computer},
  url = {https://huggingface.co/sambanovasystems/BLOOMChat-176B-v1}
  month = {5},
  year = {2023},
  version = {1.0},
}
```
