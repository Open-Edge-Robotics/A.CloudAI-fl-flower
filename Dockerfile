FROM python:3.10-bullseye

WORKDIR /ws

COPY . .

RUN pip install -r requirements.txt
RUN apt update && apt install -y net-tools neovim libgl1-mesa-glx

CMD [ "/bin/bash", "-c", "tail -f /dev/null" ]