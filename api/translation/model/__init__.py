from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("api/translation/model/nllb-200/nllb-200-distilled-600M-tokenizer")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("api/translation/model/nllb-200/nllb-200-distilled-600M-model")