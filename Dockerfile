FROM ubuntu:18.04

EXPOSE 8501
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y python3 python3-pip sudo

#RUN useradd -m alexey

#RUN chown -R alexey:alexey /home/alexey

COPY . /home/alexey/streamlit_cars/

#USER alexey

RUN cd /home/alexey/streamlit_cars/ && pip3 install -r requirements.txt
# make app directiry
WORKDIR /home/alexey/streamlit_cars

# copy requirements.txt
#COPY requirements.txt ./requirements.txt

# install dependencies
#RUN pip3 install -r requirements.txt

# copy all files over
#COPY . .

# cmd to launch app when container is run
CMD streamlit run streamlit.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
