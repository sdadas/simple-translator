FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

COPY requirements.txt requirements.txt
RUN apt update && apt install -y unzip sudo && pip install -r requirements.txt
ARG USERNAME=sdadas
COPY run_translator.py /app/run_translator.py
RUN groupadd -r $USERNAME  \
    && useradd -r -m -g $USERNAME $USERNAME \
    && chown -R $USERNAME:$USERNAME /app/ \
    && chmod a+x /opt/conda/bin/python
WORKDIR /app/

USER $USERNAME:$USERNAME
ENV DEFAULT_USER $USERNAME
ENTRYPOINT /opt/conda/bin/python run_translator.py