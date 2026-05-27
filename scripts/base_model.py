import os
import json
import yaml
import torch
from abc import ABC, abstractmethod
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from peft_extension import peft  # Load monkey patches for custom PEFT types

this_file_path = os.path.abspath(__file__)  # moose/scripts/base_model.py
repo_path = os.path.dirname(os.path.dirname(this_file_path))  # moose/
data_dir_path = os.path.join(repo_path, "data")
os.makedirs(data_dir_path, exist_ok=True)


class BaseModel(ABC):

    def __init__(
        self,
        task,
        model_name,
        use_quantized,
        seed,
        peft_config=None,
    ):
        self.task = task
        self.seed = seed
        self.model_name = model_name
        self.use_quantized = use_quantized
        self.peft_config = peft_config
        self.task_args = self.load_task_args(task)
        self.bnb_config = None
        self.modified_template = None

        if use_quantized:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

    # def get_datacollator(
    #     self,
    #     tokenizer,
    #     use_flash_attention=False,
    # ):
    #     data_collator = DataCollatorForLanguageModeling(
    #         pad_token_id=self.pad_token_id,
    #         completion_only_loss=False,
    #         padding_free=self.padding_free,
    #         return_position_ids=use_flash_attention,
    #     )
    #     if tokenizer.padding_side == "right":

    #     padding_free = False
    #     if completion_only:
    #         if use_flash_attention:
    #             padding_free = True
    #         return DataCollatorForCompletionOnlyLM(
    #             response_template=self.assistant_start_token,
    #             tokenizer=tokenizer,
    #             padding_free=padding_free,
    #         )
    #     else:
    #
    #
    #   return DataCollatorForSeq2Seq(tokenizer=tokenizer)

    def get_peft_config(self):

        return self.peft_config

    def get_model_and_tokenizer(
        self,
        tokenizer_path=None,
        use_flash_attention=True,
        use_safetensors=False,
        load_dtype=None,
    ):

        args = {"pretrained_model_name_or_path": self.model_name}

        if use_flash_attention:
            args["attn_implementation"] = "flash_attention_2"

        if use_safetensors:
            args["use_safetensors"] = use_safetensors

        # get the model
        if self.use_quantized:
            args["quantization_config"] = self.bnb_config
            args["torch_dtype"] = (
                torch.bfloat16 if not load_dtype else load_dtype
            )  # torch.bfloat16
        else:
            args["torch_dtype"] = torch.bfloat16 if not load_dtype else load_dtype
            # jje: temp fix for spiel support
            # args['torch_dtype'] = torch.float32 if self.peft_method=='spiel' else args['torch_dtype']

        print("using args: ", args)
        model = AutoModelForCausalLM.from_pretrained(**args)

        # Apply PEFT if config is provided
        if self.peft_config is not None:
            print("Applying PEFT model from peft_config")
            print(self.peft_config)
            model = peft.get_peft_model(model, self.peft_config)

        # get the tokenizer
        if tokenizer_path == None or tokenizer_path == "":
            tokenizer_path = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.fix_tokenizer(tokenizer)

        if tokenizer.vocab[tokenizer.pad_token] != tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
            print(f"Padding tokenizer pad token to {tokenizer.pad_token_id}")

        tokenizer.padding_side = "left"
        return model, tokenizer

    def get_data(
        self,
        tokenizer,
        max_seq_len,
        subset_sample_path="auto",
        filter_long_seq=False,
        load_from_cache_file=True,
        return_just_data=False,
        data_size=100,
        test_data_cap=5000,
        valid_data_ratio=0.2,
        combine_train_valid=False,
        swap_train_val=False,
    ):
        # part 1: dataset loading
        data_split = (
            list(self.task_args["split"].values())
            if "split" in self.task_args
            else None
        )
        data_split_keys = (
            list(self.task_args["split"].keys()) if "split" in self.task_args else None
        )

        # multiple subtasks should be merged if specified in the prompts yaml file
        if "combine_subsets" in self.task_args:
            assert (
                "path" in self.task_args
            ), "When combining subsets, need to provide the path to the dataset"
            ds = {}
            # merge the datasets by their splits
            for sub in self.task_args["combine_subsets"]:
                temp = load_dataset(
                    *self.task_args["path"],
                    sub,
                    split=data_split,  # trust_remote_code=True
                )
                temp_splits = (
                    data_split_keys
                    if data_split_keys is not None
                    else list(temp.keys())
                )  # train, valid(ation), test split across subtasks
                for idx, spl in enumerate(temp_splits):
                    if spl not in ds:
                        ds[spl] = []
                    split_key = (
                        idx if data_split_keys is not None else spl
                    )  # standardize data splits as "train, valid, test"
                    temp[split_key] = temp[split_key].add_column(
                        "category", [sub] * len(temp[split_key])
                    )
                    ds[spl].append(temp[split_key])
            data = DatasetDict({k: concatenate_datasets(ds[k]) for k in ds})
            del ds
        elif "json_file" in self.task_args:  # loading from local json file
            data = load_dataset(
                "json", data_files=self.task_args["json_file"], split=data_split
            )
        elif (
            "path" in self.task_args
        ):  # default way of loading data, usually if only one subtask
            data = load_dataset(
                *self.task_args["path"],
                split=data_split,  # trust_remote_code=True
            )
            # standardize data splits as "train, valid, test"
            if data_split is not None:
                ds = {k: [] for k in data_split_keys}
                for idx, k in enumerate(data_split_keys):
                    ds[k].append(data[idx])
                data = DatasetDict({k: concatenate_datasets(ds[k]) for k in ds})
                del ds
        else:
            raise ValueError(
                "No valid way to load data, need to provide either path or json_file in the prompts yaml file"
            )

        # filter based on value
        if "filter" in self.task_args:
            for key, value in self.task_args["filter"].items():
                for split in data:
                    data[split] = data[split].filter(
                        lambda example: example[key] == value
                    )

        # add example_id to the data and handle nested data
        for split in data:
            if "example_id" in data[split].column_names:
                data[split] = data[split].remove_columns("example_id")
            data[split] = data[split].add_column(
                name="example_id",
                column=[f"{split}_{idx}" for idx in range(data[split].num_rows)],
            )

        # part 2: making data subsets
        # load existing data subsets when provided
        load_saved_subset = False
        if subset_sample_path == "auto":
            subset_sample_path = os.path.join(
                data_dir_path,
                f"subset_idx_{self.task}_size{abs(data_size)}.json",
            )
            if os.path.exists(subset_sample_path):
                load_saved_subset = True
                with open(subset_sample_path, "r") as f:
                    split_to_example_ids = json.load(f)

                def _get_examples_by_ids(example_ids):
                    ids_by_split = {}
                    for example_id in example_ids:
                        s, i = example_id.split("_")
                        i = int(i)
                        if s not in ids_by_split:
                            ids_by_split[s] = []
                        ids_by_split[s].append(i)

                    for split in ids_by_split:
                        ids_by_split[split] = data[split].select(
                            sorted(ids_by_split[split])
                        )
                    if len(ids_by_split) == 1:
                        return list(ids_by_split.values())[0]
                    else:
                        return concatenate_datasets(list(ids_by_split.values()))

                new_data = DatasetDict(
                    {
                        split: _get_examples_by_ids(example_ids)
                        for split, example_ids in split_to_example_ids.items()
                    }
                )
                data = new_data

        if not load_saved_subset:
            assert (
                data_size >= 0
            ), "data_size should be non-negative when not loading saved subset"

            # get test data
            if "test" not in data:
                test_split = data["train"].train_test_split(0.1, seed=self.seed)
                data["train"] = test_split.pop("train")
                data["test"] = test_split.pop("test")
            if test_data_cap and data["test"].num_rows > test_data_cap:
                data["test"] = (
                    data["test"].shuffle(seed=self.seed).select(range(test_data_cap))
                )

            # get valid data
            if "valid" not in data:
                if "dev" in data:
                    data["valid"] = data.pop("dev")
                elif "validation" in data:
                    data["valid"] = data.pop("validation")
                else:
                    print(
                        f"No validation provided, using {valid_data_ratio * 100}% of train for validation"
                    )
                    train_split = data["train"].train_test_split(
                        test_size=valid_data_ratio, seed=self.seed
                    )
                    data["train"] = train_split.pop("train")
                    data["valid"] = train_split.pop("test")

            # limit data size if specified
            if data_size > 0:
                if valid_data_ratio and data["valid"].num_rows > int(
                    data_size * valid_data_ratio
                ):
                    data["valid"] = (
                        data["valid"]
                        .shuffle(seed=self.seed)
                        .select(range(int(data_size * valid_data_ratio)))
                    )

                train_size = data_size - data["valid"].num_rows
                if data["train"].num_rows > train_size:
                    data["train"] = (
                        data["train"].shuffle(seed=self.seed).select(range(train_size))
                    )

            # save the subset sample
            split_to_example_ids = {
                split: list(data[split]["example_id"]) for split in data
            }
            if subset_sample_path is not None:
                with open(subset_sample_path, "w") as f:
                    json.dump(split_to_example_ids, f)
                    print(f"Saved subset sample to {subset_sample_path}")
            else:
                print(f"No subset sample path provided, not saving")

        # swap train and validation sets if requested
        if swap_train_val:
            print("Swapping train and validation sets")
            data["train"], data["valid"] = data["valid"], data["train"]

        if combine_train_valid:
            data["train+valid"] = concatenate_datasets([data["train"], data["valid"]])
            del data["train"]
            del data["valid"]

        # part 3: data preprocessing
        if not return_just_data:

            # def apply_chat_template(example):
            #     return {
            #         "text": tokenizer.apply_chat_template(
            #             self.format_conversation(row=example),
            #             tokenize=False,
            #             add_special_token=False,
            #             add_generation_prompt=False,
            #         )
            #     }

            def as_conversation(example):
                return {
                    "messages": self.format_conversation(row=example),
                }

            data = data.map(
                as_conversation,
                load_from_cache_file=load_from_cache_file,
            )

            # filtering long sequences
            if filter_long_seq:
                # TODO: this is broken
                print("(pre-filtering) Data size:")
                for split in data:
                    print(f"  {split}: {data[split].num_rows}")

                def tokenize_example(example):
                    return tokenizer(example["text"], add_special_tokens=False)

                def filter_by_length(example):
                    return len(example["input_ids"]) < max_seq_len

                for split in data:
                    if split != "test":
                        data[split] = data[split].map(tokenize_example, batched=False)
                        data[split] = data[split].filter(filter_by_length)
                        data[split] = data[split].remove_columns(
                            ["input_ids", "attention_mask"]
                        )

            print("Data size:")
            for split in data:
                print(f"  {split}: {data[split].num_rows}")

        return data

    @staticmethod
    def load_task_args(task):
        if not os.path.exists(data_dir_path):
            data_zip_path = os.path.join(repo_path, "data.tar.gz")
            import shutil

            shutil.unpack_archive(data_zip_path, repo_path)

        task_args_path = os.path.join(repo_path, "tasks.yaml")
        with open(task_args_path, "r") as f:
            all_task_args = yaml.safe_load(f)

        return all_task_args[task]

    @abstractmethod
    def format_conversation(self, row):
        """Takes input from create_prompt but implemented at child class"""

    def create_prompt(self, row):
        instruction = self.task_args["instruction"]
        usr_input = self.task_args["user_input"]
        assistant_output = self.task_args["assistant_output"]

        # when there are multiple user inputs
        if isinstance(usr_input, dict):
            s = ""
            for k in usr_input:
                if isinstance(usr_input[k], dict):
                    s += "\n" + k + " - "
                    data_key = usr_input[k]["key_name"]
                    data_field = row[data_key]  # e.g. choices
                    fields = usr_input[k][
                        "fields"
                    ]  # e.g. subfield of choices; labels, description
                    concat_symbol = usr_input[k][
                        "concat_symbol"
                    ]  # how to concat the fields

                    if concat_symbol:
                        s += "\n"
                        for pair in zip(data_field[fields[0]], data_field[fields[1]]):
                            s += pair[0] + " " + concat_symbol + " " + pair[1] + "\n"
                    else:
                        s += data_field[fields[0]]
                else:
                    s += k + " - " + str(row[usr_input[k]]) + " [SEP] \n"

            usr_query = s.strip().strip("\n")[: -len("[SEP]")].strip()
        else:
            usr_query = row[usr_input]

        return instruction, usr_query, str(row[assistant_output])
