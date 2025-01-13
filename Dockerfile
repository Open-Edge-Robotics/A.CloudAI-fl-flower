FROM python:3.12

WORKDIR /ws

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /ws/src

EXPOSE 8080

CMD [ "tail", "-f", "/dev/null" ]
