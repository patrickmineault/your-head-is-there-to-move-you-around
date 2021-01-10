FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN git clone https://github.com/facebookresearch/slowfast.git; cd SlowFast; python setup.py build develop; cd ..

RUN git clone https://github.com/patrickmineault/research_code.git; cd research_code; pip install -e .; cd ..

RUN apt-get update && apt-get install -y wget unzip