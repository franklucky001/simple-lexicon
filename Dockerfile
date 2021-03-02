FROM tensorflow/tensorflow:1.14.0-gpu-py3
RUN mkdir -p /root/simple-lexicon
WORKDIR /root/simple-lexicon
COPY config config
COPY sequence_model sequence_model
COPY utils utils
ADD main.py .
ADD start_entrypoint.sh .
ADD requirements.txt .
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade pip
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com -r requirements.txt
ENTRYPOINT ["/bin/bash", "/root/simple-lexicon/start_entrypoint.sh"]

