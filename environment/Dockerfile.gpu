FROM nvidia/cuda:8.0-cudnn5-devel

# Pick up some TF dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-pip \
        git \
        unzip \
        wget \
        python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/model

COPY prepare_data.sh /tmp/prepare_data.sh

RUN chmod 0755 /tmp/prepare_data.sh \
 && /tmp/prepare_data.sh

COPY requirements_gpu.txt /tmp/requirements_gpu.txt

RUN pip3 install --upgrade pip \
 && pip3 install -U setuptools \
 && pip3 --no-cache-dir install -r /tmp/requirements_gpu.txt

WORKDIR "/tmp/model"
