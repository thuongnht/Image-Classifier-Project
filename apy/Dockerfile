FROM python:3.11-slim

RUN echo $MAPPING_PORT

WORKDIR /opt/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE $MAPPING_PORT
CMD ["python3", "app.py"]