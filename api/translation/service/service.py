from api.translation.model import Model

ml = Model()
tokenizer = ml.tokenizer
model = ml.model

def translate(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    lang = "kor_Hang"
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang), max_length=30
    )

    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result