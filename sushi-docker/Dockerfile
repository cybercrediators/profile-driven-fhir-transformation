FROM node:latest
WORKDIR /opt/fsh_dir
RUN apt update
RUN apt install nodejs npm ruby ruby-dev openjdk-21-jre -y
RUN npm install -g fsh-sushi
RUN gem install jekyll
