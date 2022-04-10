FROM python:3.8
LABEL maintaines="Massimiliano Pronesti"
LABEL repository="qlearnkit"

ENV PYTHONUNBUFFERED 1
WORKDIR /qlearnkit

COPY requirements.txt /qlearnkit/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    pennylane \
    torch


COPY . /qlearnkit
RUN pip install .[pennylane] 
 
ENV PYTHONPATH "${PYTHONPATH}:/qlearnkit"
CMD [ "python3"]

