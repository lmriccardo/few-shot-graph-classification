FROM python:3.10-slim-buster

RUN groupadd -r fsgc && useradd -r -g fsgc-group fscg
RUN chsh root -s /usr/sbin/nologin
ENV HOME /home/fscg

WORKDIR /home/fscg/