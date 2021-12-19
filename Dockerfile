FROM python:3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN mkdir -p /scripts
COPY requirements.txt assignment-5.sql assignment-5.py the_bash_script.sh midterm.py cat_cat.py cont_cat.py cont_cont.py /scripts/
WORKDIR /scripts

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /scripts
#USER appuser

RUN chmod +x the_bash_script.sh
RUN ./the_bash_script.sh
