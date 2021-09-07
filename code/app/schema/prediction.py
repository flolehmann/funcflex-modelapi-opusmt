from typing import Union, Optional
from enum import Enum
from pydantic import BaseModel


class FunctionEnum(str, Enum):
    translation_de_en = 'translation_de_en'
    translation_fr_en = 'translation_fr_en'


class PredictionBase(BaseModel):
    input: Union[dict, list, set, float, int, str, bytes, bool]
    function: FunctionEnum


class PredictionInput(PredictionBase):
    pass


class PredictionOutput(BaseModel):
    prediction: Union[dict, list, set, float, int, str, bytes, bool]
    function: Optional[FunctionEnum]
    pass
