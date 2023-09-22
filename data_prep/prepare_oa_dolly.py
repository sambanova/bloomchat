"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the BLOOMChat License, Version 1.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://github.com/sambanova/bloomchat/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
governing permissions, limitations and restrictions under the License.
"""

import json
from datasets import load_dataset


data = load_dataset("OpenAssistant/oasst1")

id_to_item = {}

for item in data["train"]:
    m_id = item["message_id"]
    id_to_item[m_id] = item

for m_id in id_to_item:
    item = id_to_item[m_id]
    if item["parent_id"] is not None:
        if "child_ids" in id_to_item[item["parent_id"]]:
            id_to_item[item["parent_id"]]["child_ids"].append(m_id)
        else:
            id_to_item[item["parent_id"]]["child_ids"] = [m_id]


def get_best_route(item):

    if "child_ids" not in item or len(item["child_ids"]) == 0:
        return [None]

    def _key(item):
        emojis = item["emojis"]
        if emojis is None:
            return 0

        score = 0
        if "+1" in emojis["name"]:
            score += emojis["count"][emojis["name"].index("+1")]
        if "-1" in emojis["name"]:
            score -= emojis["count"][emojis["name"].index("-1")]

        return score

    children = [id_to_item[child_id] for child_id in item["child_ids"]]
    selected_child = max(children, key=_key)

    return [selected_child] + get_best_route(selected_child)


selected_all = []

for m_id in id_to_item:
    item = id_to_item[m_id]
    if item["parent_id"] is not None:
        continue

    selected_all.append([item] + get_best_route(item))


texts = []

for dialogue in selected_all:
    text = ""
    for item in dialogue:
        if item is None:
            continue
        if item["role"] == "assistant":
            text += f"\n<bot>: {item['text']}"
        elif item["role"] == "prompter":
            text += f"\n<human>: {item['text']}"
        else:
            assert False

    text = text.strip()

    texts.append(text)


data = load_dataset("databricks/databricks-dolly-15k")

for item in data["train"]:
    if item["context"] == "":
        text = f"<human>: {item['instruction']}\n<bot>: {item['response']}"
    else:
        text = f"<human>: {item['instruction']}\n{item['context']}\n<bot>: {item['response']}"

    texts.append(text)


with open("oasst1_dolly.jsonl", "w") as f:
    for text in texts:
        f.write(json.dumps({"prompt": "", "completion": text}) + "\n")
