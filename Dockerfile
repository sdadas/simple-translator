FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
COPY requirements.txt requirements.txt
RUN apt update && apt install -y unzip sudo && pip install -r requirements.txt
COPY run_translator.py /app/run_translator.py
WORKDIR /app/
ENTRYPOINT /opt/conda/bin/python run_translator.py