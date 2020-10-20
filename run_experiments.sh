#!/bin/bash
SESSION='skip-avsr'
dataset='mvlrs_v1'
tmux new-session -d -s $SESSION

counter=0
while [ $counter -le 7 ]
do
  tmux new-window -t $SESSION:$((counter + 1)) -n 'gpu_'$counter;
#  tmux send-keys 'python3 experiment_lrs2.py -d '$dataset' -g '$counter C-m;
  ((counter++))
done
