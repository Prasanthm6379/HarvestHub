FROM python:3.12

WORKDIR /BackenApi

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]