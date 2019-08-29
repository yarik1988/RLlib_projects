#!/bin/bash
apt install gnome-core
apt install freeglut3-dev
apt install python3-pip

apt install libexo-1-0

pip3 install gym
pip3 install ray[rllib]
pip3 install ray[debug]
