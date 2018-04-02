# Prepare environment using pip

## CPU
`pip install -r  requirements.txt`

## GPU
`pip install -r requirements_gpu.txt`

# Prepare docker container (RECOMMENDED)

## Build container

### CPU
`docker build -t "keras-fine-tune-inception:cpu" . -f Dockerfile`
### GPU
`docker build -t "keras-fine-tune-inception:gpu" . -f Dockerfile.gpu`

## Run TRAIN container
* Replace `LOCAL_REPO_DIRECTORY` with path to the directory this repo was cloned to.
* From inside container `cpu` or `gpu` container run `jupyter notebook --ip='*'`.

### GPU
`nvidia-docker run -i -t -v LOCAL_REPO_DIRECTORY:/tmp/model -p 8888:8888 -p 6006:6006 keras-fine-tune-inception:gpu`
### CPU
`docker run -i -t -v LOCAL_REPO_DIRECTORY:/tmp/model -p 8888:8888 -p 6006:6006 keras-fine-tune-inception:cpu`

