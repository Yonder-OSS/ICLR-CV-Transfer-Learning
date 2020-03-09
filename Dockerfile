#FROM tensorflow/tensorflow:1.3.0-py3
FROM tensorflow/tensorflow:1.10.0 
# Python 2??

# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root
WORKDIR $HOME

COPY . $HOME/
RUN pip install -r $HOME/requirements.txt
RUN pip install -e .
