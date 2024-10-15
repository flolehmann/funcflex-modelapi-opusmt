FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./app/requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY ./app /app

# setup python logging environment
RUN mkdir /var/log/model-api
RUN cat >log.log
