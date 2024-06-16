from dataclasses import dataclass, field

import datasets
import torch
import transformers
import trl


QUESTION = " <Question>"
CHOICES = " <Choices>"
ANSWER = " <Answer>"


@dataclass
class CustomArguments:
    model_path: str
    data_path: str = field(default="/shared/public/data/mmlu")
    max_seq_length: int = field(default=512)
    test_split_size: float = field(default=0.05)



def formatting_func(example):
    output_texts = []
    for i in range(len(example["question"])):
        choices = ""
        for j in range(len(example["choices"][i])):
            choices += f"{j+1}. {example['choices'][i][j]}; "
        s = """Below is a question and multiple choice answers, choices separated by a semicolon. \
Please select the best answer for the question. """
        s += f"{QUESTION}{example['question'][i]} "
        s += f"{CHOICES}{choices} "
        s += f"{ANSWER}{example['answer'][i]}"
        output_texts.append(s)
    return output_texts


def main():
    parser = transformers.HfArgumentParser(
        (transformers.TrainingArguments, CustomArguments)
    )
    training_args, custom_args = parser.parse_args_into_dataclasses()


    dataset = datasets.load_from_disk(custom_args.data_path)[
        "auxiliary_train"
    ].train_test_split(test_size=custom_args.test_split_size, seed=training_args.seed)
    dataset_train, dataset_eval = dataset["train"], dataset["test"]


    if torch.distributed.get_rank() == 0:
        print(custom_args, training_args)


    model = transformers.AutoModelForCausalLM.from_pretrained(
        custom_args.model_path,
        use_cache=False,
        attn_implementation="sdpa",
    )

    if torch.distributed.get_rank() == 0:
        print("model dtype", model.dtype)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        custom_args.model_path, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    answer_token_id = tokenizer.encode(ANSWER, add_special_tokens=False)
    collator = trl.DataCollatorForCompletionOnlyLM(answer_token_id, tokenizer=tokenizer)


    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        formatting_func=formatting_func,
        max_seq_length=custom_args.max_seq_length,
        args=training_args,
        data_collator=collator
    )
    trainer.train()


if __name__ == "__main__":
    main()
