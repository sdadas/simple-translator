mkdir -p ./model_cache
mkdir -p ./sentence_cache
docker run -d -p 80:8080 -v "$(pwd)/model_cache":/app/model_cache -v "$(pwd)/sentence_cache":/app/sentence_cache --user "$(id -u):$(id -g)" translator