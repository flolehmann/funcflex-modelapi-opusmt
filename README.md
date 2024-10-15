# Opus-MT model API, as part of the paper "Functional Flexibility in Generative AI Interfaces"

This model API makes GPT-neo available to carry out `translate` tasks.

See the paper on arxiv: https://arxiv.org/abs/2410.10644

## Run the API:

1. Inside the directory `code/app` create a `.env` file. Add content as shown below.
   ```
   PUBLIC_PREDICTION = False
   API_KEY = "YOUR-API-KEY"
   ```
   If `PUBLIC-PREDICTION` is set to True, authentication will be bypassed. If set to False, you need to send the `API_KEY` 
   with every HTTP request as `X-API-Key` header in order to get authenticated.
2. Choose:
   1. To run the API locally, execute `docker-compose up -d`
   2. To deploy the API, execute `docker-compose -f docker-compose.prod.yml up -d`
3. The API will be available at `http://localhost:8008/` a documentation can be found at `http://localhost:8008/docs`