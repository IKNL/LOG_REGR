# basic python3 image as base
FROM python:3

# copy all local files to the image
COPY . /

# for testing locally only. 
# Comment this line out when deploying
# ENV DATABASE_URI /app/local/database.csv

# install pytaskmanageexitri
RUN pip install ppdli


# install external dependancies into enviroment
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# execute algorithm in the container
CMD ["python", "./main.py"]