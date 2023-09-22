"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the BLOOMChat License, Version 1.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://github.com/sambanova/bloomchat/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
governing permissions, limitations and restrictions under the License.
"""

import random
from datasets import load_dataset
import json

all_items = []
data_list = [
    "unified_multi_news.jsonl",
    "unified_openai_summarize_tldr.jsonl",
    "unified_scitldr.jsonl",
    "unified_oa_v3_fixed_plus_safety_fixed.jsonl",
    "unified_soda_dialog.jsonl",
    "unified_unifiedskg_instructions_v2.jsonl",
    "unified_cot_instructions.jsonl",
    "unified_unatural_instructions.jsonl",
    "unified_squad_v2.jsonl",
    "unified_conv_finqa.jsonl",
    "unified_nq.jsonl",
    "unified_plot_screenplay_books_dialog.jsonl",
    "unified_oscar_en_sample_dialog.jsonl",
    "unified_ul2_plus_oscar_en_sample_dialog.jsonl",
    "unified_essays.jsonl",
    "unified_merged_code_xp3.jsonl",
    "unified_grade_school_math_instructions.jsonl",
    "unified_poetry_instructions.jsonl",
    "unified_chip2.jsonl",
    "unified_joke_explanations.jsonl",
    "unified_cuad.jsonl",
    "unified_p3.jsonl.gz",
    "unified_ni.jsonl.gz",
    "unified_flan.jsonl.gz",
    "unified_basic.jsonl",
]
for data_path in data_list:

    print(data_path)
    try:
        data = load_dataset("laion/OIG", data_files=data_path, streaming=True)["train"]
    except FileNotFoundError:
        print(f"Could not find {data_path} in laion/OIG dataset")
        continue

    cur_data = []
    for ii, item in enumerate(data):

        if ii >= 1000000:
            break

        if "meta" in item:
            # prosocial sometimes produce unsafe answers
            if item["meta"]["source"] == "prosocial":
                continue

        new_item = {
            "prompt": "",
            "completion": item["text"].strip() + "\n",
        }
        item = new_item

        if data_path == "unified_basic.jsonl":
            for _ in range(1000):  # mk sure basic instructions ocasionally appear
                all_items.append(item)

        all_items.append(item)
        cur_data.append(item)

random.shuffle(all_items)

with open("bloom_ock_100K_each.jsonl", "w") as f:
    for item in all_items:
        f.write(json.dumps(item) + "\n")
