FROM python:3.8
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

CMD ["python", "-u", "__init__.py"]