FROM hub.fuxi.netease.com/danlu/ubuntu:18.04_py3_cpu
FROM python:3.7
#FROM continuumio/anaconda3

ARG REDIS_VERSION=6.0.9

# 请从此处开始编写
RUN apt-get update && apt-get install -y \
    zsh \
    tmux \
    cmake \
    make \
    openssh-server \
    openssh-sftp-server \
    net-tools \
    sudo \
    vim \
    screen \
    unzip \
    screen \
    iftop \
    dos2unix \
    build-essential \
    curl \
    vim \
    wget \
    libopencv-dev \
    libsnappy-dev \
    python-dev \
    python-pip \
    google-perftools \
    git-core && \
    rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh \
    && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
    && chsh -s /bin/zsh

RUN mkdir -p /etc/redis && mkdir -p /var/redis/6379  && \
    wget -c http://download.redis.io/releases/redis-$REDIS_VERSION.tar.gz && \
	tar -zxvf redis-$REDIS_VERSION.tar.gz && \
    cd ./redis-$REDIS_VERSION && \
    make && \
    cp ./src/redis-server /usr/local/bin/redis-server && \
    cp ./src/redis-cli /usr/local/bin/redis-cli && \
    cp ./utils/redis_init_script /etc/init.d/redis_6379 && \
    sed -i "s/daemonize no/daemonize yes/g" ./redis.conf && \
    sed -i "s/bind 127.0.0.1/bind 0.0.0.0/g" ./redis.conf && \
    sed -i 's/loglevel notice/loglevel verbose/g' ./redis.conf && \
    sed -i 's/logfile \"\"/logfile \"\/var\/log\/redis_6379.log"/g' ./redis.conf && \
    sed -i 's/dir .\//dir \/var\/redis\/6379/g' ./redis.conf && \
    cp ./redis.conf /etc/redis/6379.conf && \
    update-rc.d redis_6379 defaults && \
    cd .. && \
    rm -rf ./redis-$REDIS_VERSION.tar.gz ./redis-$REDIS_VERSION

# Add conda path
#RUN echo export PATH="/opt/conda/bin:$PATH" >> /root/.bash_profile
# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all



RUN apt-get clean && apt-get update
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
RUN sudo apt-get install -y nodejs

RUN pip install jupyterlab logbook jupyterlab_github tensorflow==2.3.0 torch pybullet \
    akshare==0.9.78 \
    backtrader==1.9.76.123 \
    beautifulsoup4==4.9.3 \
    bidict==0.21.2 \
    Cerberus==1.3.4 \
    cowpy==1.1.0 \
    docxtpl==0.11.5 \
    easydict==1.9 \
    elo==0.1.1 \
    Flask==2.0.1 \
    flask_socketio==5.1.0 \
    imageio==2.9.0 \
    Jinja2==3.0.1 \
    keras==2.4.3 \
    line_profiler==3.3.0 \
    Logbook==1.5.3 \
    lxml==4.6.3 \
    matplotlib==3.4.2 \
    mock==4.0.3 \
    msgpack_python==0.5.6 \
    munch==2.5.0 \
    numpy==1.18.5 \
    paddle==1.0.2 \
    pandas==1.2.4 \
    pandas_profiling==3.0.0 \
    pdfkit==0.6.1 \
    pipreqs \
    Pillow==8.2.0 \
    psutil==5.8.0 \
    pyautogui==0.9.52 \
    pyecharts==1.9.0 \
    pyglet==1.5.17 \
    pyparsing==2.4.7 \
    PyPDF2==1.26.0 \
    pytest==6.2.4 \
    PyYAML==5.4.1 \
    recordclass==0.14.3 \
    recordtype==1.3 \
    requests==2.25.1 \
    rpyc==5.0.1 \
    selenium==3.141.0 \
    stable_baselines==2.10.2 \
    statemachine==0.1 \
    statsmodels==0.12.2 \
    sympy==1.8 \
    tensorflow==2.3.0 \
    tensorflow_probability==0.13.0 \
    torch==1.9.0 \
    torchvision==0.10.0 \
    trueskill==0.4.5 \
    zerorpc==0.6.3



#RUN conda install ipykernel
#RUN conda create -n jupyter python=3.7 ipykernel jupyterlab
#RUN conda create --name rlease python=3.7


# Install Rust development environment
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"
