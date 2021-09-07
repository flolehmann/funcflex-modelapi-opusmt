import torch
from decouple import config
from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction

from starlette import status

from definitions import MODEL_DIR

from transformers import MarianTokenizer, MarianMTModel

router = APIRouter()

ENTITY = "Machine Learning"

STAGE = config("STAGE")

# load model only once:

models = {}
tokenizers = {}

if STAGE == "PROD":
    device = torch.device('cuda')
    models[schema.prediction.FunctionEnum.translation_de_en] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en').to(device)
    models[schema.prediction.FunctionEnum.translation_fr_en] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en').to(device)
else:
    models[schema.prediction.FunctionEnum.translation_de_en] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    models[schema.prediction.FunctionEnum.translation_fr_en] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')

tokenizers[schema.prediction.FunctionEnum.translation_de_en] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
tokenizers[schema.prediction.FunctionEnum.translation_fr_en] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')


@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):

    model_function = data.function

    print(model_function)

    model = models[model_function]
    tokenizer = tokenizers[model_function]

    preprocess_text = data.input.strip().replace("\n", "")
    opus_prepared_text = [preprocess_text]

    if STAGE == "PROD":
        batch = tokenizer(opus_prepared_text, return_tensors="pt").to(device)
    else:
        batch = tokenizer(opus_prepared_text, return_tensors="pt")

    generator = model.generate(**batch)
    translated = tokenizer.batch_decode(generator, skip_special_tokens=True)

    return {
        "prediction": translated,
        "function": model_function
    }


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
