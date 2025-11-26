#!/bin/bash

echo "123" | sudo -S chmod 777 /dev/ttyAM0 
cd /home/hmkk/car_ws

gnome-terminal -- bash -c 'source /home/hmkk/car_ws/car_env/bin/activate && python3 main.py; exec bash'