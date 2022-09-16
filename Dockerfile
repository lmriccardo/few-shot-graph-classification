FROM ubuntu:22.10
LABEL org.opencontainers.image.authors="riccardo.lamarca98@gmail.com"

# Install python3.10.7 and python PIP
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y python3.10 && ln -sf python3 /usr/bin/python
RUN apt-get install -y python3-pip && ln -sf pip3 /usr/bin/pip

# Install required packages
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir torch-scatter     \
                               torch-sparse      \
                               torch-cluster     \
                               torch-spline-conv \
                               torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip install --no-cache-dir numpy matplotlib plotly networkx sklearn wrapt rich

# Add a new no-sudo user and disable root login
RUN groupadd -r fsgc-group && useradd -r -g fsgc-group fsgc
RUN chsh -s /usr/sbin/nologin root

# Set some environment variables
ENV LANG en_US.utf8
ENV HOME /home/fscg

# Set the working directory and copy all required files
COPY --chown=fsgc:fsgc-group ./models/ $HOME/app/models/
COPY --chown=fsgc:fsgc-group ./src/ $HOME/app/src/
COPY --chown=fsgc:fsgc-group ./data/ $HOME/app/data/
WORKDIR $HOME/app/src/

# Set the user to use inside the container
USER fsgc:fsgc-group

ENTRYPOINT ["uname", "-r"]