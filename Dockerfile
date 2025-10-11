FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

COPY ./pyarrow.sh ./
RUN chmod +x ./pyarrow.sh

ENV HTTP_PROXY=http://proxy11.omu.ac.jp:8080/
ENV HTTPS_PROXY=http://proxy11.omu.ac.jp:8080/
ENV http_proxy=http://proxy11.omu.ac.jp:8080/
ENV https_proxy=http://proxy11.omu.ac.jp:8080/

ENV LANG="ja_JP.UTF-8" \
    LANGUAGE="ja_JP:ja" \
    LC_ALL="ja_JP.UTF-8"

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y curl git zip unzip make cmake python3-pip
RUN ./pyarrow.sh

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

ENV PATH="/root/.local/bin:${PATH}"

RUN uv tool install mlflow
RUN uv tool install 'dvc[all]'


CMD ["sh"]
