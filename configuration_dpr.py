from transformers import PretrainedConfig, AutoConfig
import transformers
from typing import Union, Dict, Any


class CustomDPRContrastiveConfig(PretrainedConfig):
    def __init__(
        self,
        ctx_config: Union[str, PretrainedConfig, dict] = None,
        question_config: Union[str, PretrainedConfig, dict] = None,
        ctx_cofnig_kwagrs: Dict[str, Any] = {},
        question_config_kwagrs: Dict[str, Any] = {},
        projection_dim: int = 0,
        initializer_range: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(ctx_config, str):
            ctx_config = AutoConfig.from_pretrained(ctx_config, **ctx_cofnig_kwagrs)
            ctx_name = ctx_config.__class__.__name__
            setattr(ctx_config, "class_name", ctx_name)
            setattr(ctx_config, "projection_dim", projection_dim)

        elif isinstance(ctx_config, dict):
            # save_pretrained된 checkpoint된 모델을 불러올 때 config값이 dict 형태로 들어오기 때문에 dict를 처리하는 구간이 필요로 함.
            # 문제는 PretrainedConfig.from_dict을 하면 class가 PretrainedConfig가 되어 버리기 때문에 AutoModel.from_config를 하는 도중 weight를 정상적으로 불러 들이지 못하는 문제가 발생함.
            # ctx_config = PretrainedConfig.from_dict(ctx_config)
            ctx_config = getattr(transformers, ctx_config["class_name"])(**ctx_config)

        if isinstance(question_config, str):
            question_config = AutoConfig.from_pretrained(question_config, **question_config_kwagrs)
            question_name = question_config.__class__.__name__
            setattr(question_config, "class_name", question_name)
            setattr(question_config, "projection_dim", projection_dim)

        elif isinstance(question_config, dict):
            # question_config = PretrainedConfig.from_dict(question_config)
            question_config = getattr(transformers, question_config["class_name"])(
                **question_config
            )

        self.ctx_config = ctx_config
        self.question_config = question_config

        self.projection_dim = projection_dim
        self.initializer_range = initializer_range
