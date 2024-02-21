import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from setproctitle import setproctitle
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer, TrainingArguments
from transformers import logging as hf_logging
from transformers import set_seed
from transformers.data.data_collator import DataCollatorMixin, InputDataClass

from configuration_dpr import CustomDPRContrastiveConfig
from modeling_dpr import DPRForContrastive
from trainer import DPRTrainer

hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

os.environ["TORCHDYNAMO_DISABLE"] = "1"


@dataclass
class DPRCollator(DataCollatorMixin):
    ctx_tokenizer: PreTrainedTokenizer
    question_tokenizer: PreTrainedTokenizer
    return_tensors: str = "pt"

    def torch_call(self, features: List[InputDataClass]) -> Dict[str, Any]:
        context_input = [{"input_ids": feature["context_input_ids"]} for feature in features]
        question_input = [{"input_ids": feature["question_input_ids"]} for feature in features]

        context_batch = self.ctx_tokenizer.pad(
            context_input,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
            padding=True,
        )
        question_batch = self.question_tokenizer.pad(
            question_input,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
            padding=True,
        )

        batch = dict()
        batch["ctx_input_ids"] = context_batch["input_ids"]
        batch["ctx_attention_mask"] = context_batch["attention_mask"]
        batch["question_input_ids"] = question_batch["input_ids"]
        batch["question_attention_mask"] = question_batch["attention_mask"]
        batch["return_loss"] = True

        return batch


def main(train_args: TrainingArguments) -> None:
    def preprocess(examples):
        # deberta는 cls, sep(eos) 자동으로 넣어줌
        context_outputs = tokenizer(
            tokenizer.cls_token + examples["context"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        question_outputs = tokenizer(
            tokenizer.cls_token + examples["question"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        batch = dict()
        batch["context_input_ids"] = context_outputs["input_ids"]
        batch["question_input_ids"] = question_outputs["input_ids"]

        return batch

    def compute_metrics(pred):
        pred_logits = torch.tensor(pred.predictions)
        target = torch.arange(pred_logits.size(0))

        acc = (pred.detach().cpu().max(1).indices == target).sum().float() / pred_logits.size(0)
        return {"accuracy": acc}

    config = CustomDPRContrastiveConfig(
        ctx_config="team-lucid/deberta-v3-base-korean",
        question_config="team-lucid/deberta-v3-base-korean",
        gradient_checkpointing=train_args.gradient_checkpointing,
    )
    model = DPRForContrastive(config)
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")

    dataset = load_dataset("json", data_files={"train": "/root/clip/korquad_klue_bm25idx.json"})

    with train_args.main_process_first(desc="train_data_preprocess"):
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            preprocess,
            num_proc=1,
            batched=False,
        )

    collator = DPRCollator(ctx_tokenizer=tokenizer, question_tokenizer=tokenizer)
    trainer = DPRTrainer(
        model=model,
        tokenizer=tokenizer,  # TODO: 언젠간 processor를 만들어야 할 듯
        data_collator=collator,
        train_dataset=train_dataset,
        args=train_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if "__main__" in __name__:
    parser = HfArgumentParser([TrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    setproctitle(train_args.run_name or "llm_nia")
    set_seed(train_args.seed)

    main(train_args)
