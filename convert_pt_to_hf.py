from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import _load_state_dict_into_model
import torch
from copy import deepcopy


def main() -> None:
    config = "/root/clip/deberta"
    config = AutoConfig.from_pretrained(config)
    model = AutoModel.from_config(config)

    state_dict = torch.load("/root/clip/deberta/mp_rank_00_model_states.pt")
    state_dict = {
        k.replace("query_encoder.", ""): v
        for k, v in state_dict["module"].items()
        if "query_encoder" in k
    }

    check_model = deepcopy(model)
    error_msgs = _load_state_dict_into_model(model, state_dict, "")
    assert error_msgs == [], "not pass"

    for check_param, model_param in zip(check_model.named_parameters(), model.named_parameters()):
        name = check_param[0]
        check_param = check_param[1]
        model_param = model_param[1]
        # False여야 정상적으로 weight가 불러와진 거
        print(f"동일 여부: {bool((check_param == model_param).all())}, 이름: {name}")

    model.save_pretrained("/root/clip/deberta")


if "__main__" in __name__:
    main()
