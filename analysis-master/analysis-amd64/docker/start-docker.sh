cd ..
docker build -t tra-analysis-amd64-dev -f docker/Dockerfile .
docker run -it tra-analysis-amd64-dev