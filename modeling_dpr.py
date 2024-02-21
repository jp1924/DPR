import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.dpr.modeling_dpr import (
    DPRContextEncoderOutput,
    DPREncoder,
    DPRPretrainedContextEncoder,
    DPRPreTrainedModel,
    DPRPretrainedQuestionEncoder,
    DPRQuestionEncoderOutput,
)
from transformers.utils import ModelOutput

from configuration_dpr import CustomDPRContrastiveConfig


def ibn_loss(pred: torch.FloatTensor):
    """in-batch negative를 활용한 batch의 loss를 계산합니다.
    pred : bsz x bsz 또는 bsz x bsz*2의 logit 값을 가짐. 후자는 hard negative를 포함하는 경우.
    """
    bsz = pred.size(0)
    target = torch.arange(bsz, device=pred.device)  # 주대각선이 answer
    return torch.nn.functional.cross_entropy(pred, target)


@dataclass
class DPROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    ctx_embeds: torch.FloatTensor = None
    question_embeds: torch.FloatTensor = None
    ctx_outputs: DPRContextEncoderOutput = None
    question_outputs: DPRQuestionEncoderOutput = None


class DPRCustomEncoder(DPREncoder):
    # TODO: default DPR이 bert model로 되어 있지만, AutoModel을 사용할 땐 어떻게 하면 좋을 지 몰라서 일단 bert_model로 놔둠
    base_model_prefix = "bert_model"

    def __init__(self, config: PretrainedConfig):
        DPRPreTrainedModel.__init__(self, config)
        # super하고 동일한 작동을 함. (MRO순서 바꿈)
        # 다만 DPREncoder는 사용하고 싶지만 init을 하고 싶지 않을 때 이렇게 하면 됨

        # model에 유무에 따라 add_pooling_layer가 있는 녀석이 있고 없는 녀석이 있기 때문에
        # 조건에 따라 add_pooling_layer를 넣어야 하는 방법이 달라 짐
        self.config = config
        model_kwargs = {"add_pooling_layer": False} if self.check_pooling_layer(config) else {}

        # NOTE: forward에서 self.bert_model 이란 명칭을 사용하기 때문에 bert_model을 사용함.
        self.bert_model = AutoModel.from_config(config, **model_kwargs)
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")

        # TODO: 이 부분은 어떻게 해야할지 고민 해야봐야 함.
        # 모델마다 projection_dim이름이 다를건데 이걸 어떻게 통일 시킬지 고민해볼 것
        self.projection_dim = getattr(config, "projection_dim", 0)
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        # Initialize weights and apply final processing

        self.post_init()

    def check_pooling_layer(self, config) -> bool:
        # deberta와 같이 pooling_layer가 있는 녀석이 있고 없는 녀석이 있기 때문에 이를 확인해야 함.
        model_class = _get_model_class(config, AutoModel._model_mapping)
        signature = inspect.signature(model_class.__init__)

        return "add_pooling_layer" in list(signature.parameters.keys())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor, ...]]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DPRForContrastive(DPRPreTrainedModel):
    config_class = CustomDPRContrastiveConfig

    def __init__(self, config: CustomDPRContrastiveConfig):
        super().__init__(config)

        self.config = config

        self.ctx_config = config.ctx_config
        self.question_config = config.question_config

        # TODO: DPRForContrastive용 config 만들기
        self.ctx_encoder = DPRCustomEncoder(self.ctx_config)
        self.question_encoder = DPRCustomEncoder(self.question_config)

        self.post_init()

    def post_init(self) -> None:
        self.ctx_encoder.post_init()
        self.question_encoder.post_init()

    def forward(
        self,
        ctx_input_ids: Optional[torch.Tensor] = None,
        question_input_ids: Optional[torch.Tensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        ctx_token_type_ids: Optional[torch.Tensor] = None,
        question_token_type_ids: Optional[torch.Tensor] = None,
        ctx_inputs_embeds: Optional[torch.Tensor] = None,
        question_inputs_embeds: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[DPROutput, Tuple[torch.Tensor, ...]]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # NOTE: 두 인코더를 학습시키다 보니 input값을 각각 검사해야 함. DPRPretrainedQuestionEncoder, DPRPretrainedContextEncoder를 사용하면 이 문제가 해결 되지만
        # 이후 checkpoint를 불러올 때 layer이름이 맞지 않아서 불러오는데 문제가 생기거나 혹은 clip 처럼 별도의 모듈처럼 빼야 하는 문제가 발생함.
        # 그래서 DPRCustomEncoder를 다이랙트로 불러들임.
        if (ctx_input_ids is not None and ctx_inputs_embeds is not None) or (
            question_input_ids is not None and question_inputs_embeds is not None
        ):
            raise ValueError(
                "You must insert [ctx_input_ids, ctx_inputs_embeds] and [question_input_ids, question_inputs_embeds]"
            )
        elif (ctx_input_ids is not None) and (question_input_ids is not None):
            self.warn_if_padding_and_no_attention_mask(ctx_input_ids, ctx_attention_mask)
            self.warn_if_padding_and_no_attention_mask(question_input_ids, question_attention_mask)

            ctx_input_shape = ctx_input_ids.size()
            question_input_shape = question_input_ids.size()
        elif (ctx_inputs_embeds is not None) and (question_inputs_embeds is not None):
            ctx_input_shape = ctx_inputs_embeds.size()[:-1]
            question_input_shape = question_inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either ctx_input_ids or ctx_inputs_embeds or question_input_ids or question_inputs_embeds"
            )

        device = ctx_input_ids.device if ctx_input_ids is not None else ctx_inputs_embeds.device

        if ctx_attention_mask is None:
            ctx_attention_mask = (
                torch.ones(ctx_input_shape, device=device)
                if ctx_input_shape is None
                else (ctx_input_shape != self.config.pad_token_id)
            )
        if question_attention_mask is None:
            question_attention_mask = (
                torch.ones(question_input_shape, device=device)
                if question_input_shape is None
                else (question_input_shape != self.config.pad_token_id)
            )

        if ctx_token_type_ids is None:
            ctx_token_type_ids = torch.zeros(ctx_input_shape, dtype=torch.long, device=device)

        if question_token_type_ids is None:
            question_token_type_ids = torch.zeros(
                question_input_shape, dtype=torch.long, device=device
            )

        ctx_outputs = self.ctx_encoder(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask,
            token_type_ids=ctx_token_type_ids,
            inputs_embeds=ctx_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ctx_embeds = ctx_outputs[1]

        # TODO: clip 처럼 projection layer를 추가할지 말지 고민해 봐야 할 듯

        question_outputs = self.question_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            token_type_ids=question_token_type_ids,
            inputs_embeds=question_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        question_embeds = question_outputs[1]

        logits = torch.matmul(question_embeds, ctx_embeds.t())

        loss = None
        if return_loss:
            loss = ibn_loss(logits)

        if not return_dict:
            output = (
                logits,
                ctx_embeds,
                question_embeds,
                ctx_outputs,
                question_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return DPROutput(
            loss=loss,
            logits=logits,
            ctx_embeds=ctx_embeds,
            question_embeds=question_embeds,
            ctx_outputs=ctx_outputs,
            question_outputs=question_outputs,
        )


class DPRCustomContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = getattr(config, "ctx_config", config)

        # NOTE: 마개조 마개조 마개조 마개조 마개조 마개조 마개조 마개조 마개조
        # config가 dict로 들어오는 경우 post_init을 할 때 애러가 발생함.
        # 그리고 PretrainedConfig로 불러들이면 _get_model_class(config, AutoModel._model_mapping), AutoModel할 때 정상적으로 불러들이지 못할 가능성이 있음.
        if isinstance(self.config, dict):
            self.config = getattr(transformers, self.config["class_name"]).from_dict(self.config)

        self.ctx_encoder = DPRCustomEncoder(self.config)

        self.post_init()


class DPRCustomDPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = getattr(config, "question_config", config)

        # NOTE: 마개조 마개조 마개조 마개조 마개조 마개조 마개조 마개조 마개조
        # config가 dict로 들어오는 경우 post_init을 할 때 애러가 발생함.
        # 그리고 PretrainedConfig로 불러들이면 _get_model_class(config, AutoModel._model_mapping), AutoModel할 때 정상적으로 불러들이지 못할 가능성이 있음.
        if isinstance(self.config, dict):
            self.config = getattr(transformers, self.config["class_name"]).from_dict(self.config)

        self.question_encoder = DPRCustomEncoder(self.config)

        self.post_init()


def test() -> None:
    config = CustomDPRContrastiveConfig(
        ctx_config="klue/roberta-base",
        question_config="klue/bert-base",
    )
    model = DPRForContrastive(config)

    model.save_pretrained("dpr_save_test", safe_serialization=False)
    model.from_pretrained("dpr_save_test")

    DPRCustomContextEncoder.from_pretrained("dpr_save_test")
    DPRCustomDPRQuestionEncoder.from_pretrained("dpr_save_test")


if "__main__" in __name__:
    test()
    # import json

    # with open("/root/clip/korquad_klue_bm25_sampler_indices.json", "r") as f:
    #     data = json.load(f)
