"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the BLOOMChat License, Version 1.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://github.com/sambanova/bloomchat/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
governing permissions, limitations and restrictions under the License.
"""
import argparse
import json

from generative_tuning.generative import GenerativePipeline


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Model args")
    parser.add_argument(
        "--pef",
        default=None,
        type=str,
        required=True,
        help="PEF",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Name of model or path to model. Used in initialize of model class.",
    )
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache directory for models.")
    args = parser.parse_args()
    return args


def main() -> None:
    """Run main."""
    kwargs = parse_args()
    model_name_or_path = kwargs.model_name_or_path
    if not model_name_or_path:
        raise ValueError("Must provide model_name_or_path.")

    generator = GenerativePipeline(kwargs.model_name_or_path, kwargs.pef, cache_dir=kwargs.cache_dir)

    while True:
        input_text = input("User Input Text: ")

        input_text = input_text.strip()
        modified_input_text = f"<human>: {input_text}\n<bot>:"

        output = generator.predict(
            modified_input_text,
            max_tokens_to_generate=512,
            repetition_penalty=1.2,
            top_p=0.9,
            return_completion_only=True,
            do_sample=False,
        )[0]
        completion = output["text"]

        print(f"Model output text: {completion}")


if __name__ == "__main__":
    main()
