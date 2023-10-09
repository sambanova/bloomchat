# Copyright 2023 SambaNova Systems, Inc.

# Licensed under the BLOOMChat License, Version 1.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

# https://github.com/sambanova/bloomchat/blob/main/LICENSE

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions, limitations and restrictions under the License.

#!/bin/bash
DATA_PREP_PYTHONPATH=$1
PYTHONPATH=$DATA_PREP_PYTHONPATH:$PYTHONPATH python -m generative_data_prep pipeline --input_file_path=../data_prep/bloom_ock_100K_each.jsonl --output_path=./bloom_ock_100k_out/ --dev_ratio=0.0 --test_ratio=0.0 --shuffle on_RAM --pretrained_tokenizer=bigscience/bloom --max_seq_length=2048  --input_packing_config greedy --num_training_splits=8
