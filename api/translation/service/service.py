from transformers import pipeline
from api.translation.model import Model
from api.translation.response.response import Languages
from api.translation.constants.constants import Language

ml = Model()
tokenizer = ml.tokenizer
model = ml.model
model.to('cuda:0')


def translate(text: str):
    result = Languages()
    setattr(result, "origin", text)
    for lang in Language:
        setattr(result, lang.name.lower(), predict(text, lang.value))
    return result


def predict(text: str, code: str):
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer,
                                    src_lang='', tgt_lang=code, device=0)
    return translation_pipeline.predict(text)[0].get("translation_text")
