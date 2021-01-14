FROM python:3.8
COPY requirements.txt .
RUN python -m pip install requirements.txt