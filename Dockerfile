FROM python:3.8
#RUN apt-get update && apt install -y vim wget curl unzip w3m zsh && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /home/khorav/sann
COPY requirements.txt ./
RUN pip install --no-cache-dir -r reqs.txt
COPY . ./
