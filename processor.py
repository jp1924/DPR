from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding


class DPRProcessor(ProcessorMixin):
    attributes = ["ctx_tokenizer", "question_tokenizer"]

    def __init__(self, ctx_tokenizer, question_tokenizer, **kwargs):
        super().__init__(ctx_tokenizer, question_tokenizer, **kwargs)

    def __call__(self, ctx_text=None, question_text=None, return_tensors=None, **kwargs):
        """ """

        if ctx_text is None and question_text is None:
            raise ValueError(
                "You have to specify either ctx_text or question_text. Both cannot be none."
            )

        if ctx_text is not None:
            ctx_encoding = self.ctx_tokenizer(ctx_text, return_tensors=return_tensors, **kwargs)

        if question_text is not None:
            question_encoding = self.question_tokenizer(
                question_text, return_tensors=return_tensors, **kwargs
            )

        if ctx_text is not None and question_text is not None:
            ctx_encoding["pixel_values"] = question_encoding.input_ids
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
