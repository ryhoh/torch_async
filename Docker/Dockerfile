FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# 開発時によく使うソフトウェアをインストール
RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  wget \
  curl \
  vim \
  git \
  locales \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# 日本語化
RUN locale-gen ja_JP.UTF-8 \
  && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# pythonのシンボリックリンクを張り替え
RUN ln -s /usr/bin/python3 /usr/bin/python
# pipのシンボリックリンクを張り替え
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# python用のライブラリをインストール
COPY requirements.txt /
RUN pip install -r requirements.txt
