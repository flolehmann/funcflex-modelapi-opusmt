from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware

from api.v1 import machine_learning

#logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
#logger = logging.getLogger('model-api')

API_PREFIX = "/api/v1"

app = FastAPI(title="model-api-opusmt", version="1")

origins = [
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    machine_learning.router,
    prefix=API_PREFIX,
    tags=["Machine Learning"]
)
