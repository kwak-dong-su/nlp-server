from api.translation.model import Model
from api.translation.response.response import Languages
from api.translation.constants.constants import Language

ml = Model()
tokenizer = ml.tokenizer
model = ml.model

def translate(text: str):
    result = Languages()
    setattr(result, "origin", text)
    for lang in Language:
        setattr(result, lang.name.lower(), predict(text, lang.value))
    return result

def predict(text: str, code: str):
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(code), max_length=30
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]