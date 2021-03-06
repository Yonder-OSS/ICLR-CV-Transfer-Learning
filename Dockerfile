FROM tensorflow/tensorflow:1.10.0-gpu

# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root
WORKDIR $HOME

COPY . $HOME/
RUN pip install -r $HOME/requirements.txt
RUN pip install -e .

CMD ["python", "/root/train.py", "/root/configs/train.json"]
#CMD ["python", "/root/eval.py", "/root/configs/eval.json"]