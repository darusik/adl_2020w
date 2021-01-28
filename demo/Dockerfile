FROM python:3.7-stretch

RUN git clone https://github.com/cleardusk/3DDFA_V2.git /tmp/3DDFA_V2 && \
    cd /tmp/3DDFA_V2/ && \
    git checkout 70e7cde

RUN apt-get update && apt-get install -y --no-install-recommends rsync libgl1-mesa-glx
RUN pip install notebook==6.1.5 ipywidgets==7.5.1 voila==0.2.4

# NOTE: Uncomment to build faster when make several iterative
#       changes to something in adl_2020 dir
#       Otherwise, it will be installed from requirements.txt later
RUN pip install torch==1.5.1

COPY . /3DDFA_V2
WORKDIR /3DDFA_V2

RUN rsync -a /tmp/3DDFA_V2/ .
RUN (patch < requirements.patch && pip install -r requirements.txt && sh ./build.sh)

CMD ["voila", "Mask-Demo.ipynb"]