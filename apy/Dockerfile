FROM python:3.11-slim

RUN echo $MAPPING_PORT

WORKDIR /opt/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY static static
COPY cat_to_name.json cat_to_name.json
COPY predict.py predict.py
COPY app.py app.py

EXPOSE $MAPPING_PORT
CMD ["python3", "app.py"]