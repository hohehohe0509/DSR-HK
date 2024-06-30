FROM thebloke/cuda11.8.0-ubuntu22.04-pytorch:latest

MAINTAINER hohehohe

WORKDIR /app

RUN pip install scikit-learn==1.4.2
RUN pip install torch==2.0.1+cu118
RUN pip install numpy==1.26.2
RUN pip install scipy==1.11.4
RUN pip install numba==0.58.1
RUN pip install tqdm
RUN pip install pandas==2.1.3
RUN pip install pyyaml
RUN pip install pydantic
RUN pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

COPY ./ ./