SESSION=skip-avsr
dataset='mvlrs_v1'
num_gpus=8
tmux new-session -d -s $SESSION
counter=1
while [ $counter -le $num_gpus ]
do
  tmux new-window -t $SESSION:$counter -n 'gpu_'$counter
  tmux send-keys 'python experiment_lrs2.py -d '$dataset' -g '$counter C-m
  ((counter++))
done

tmux attach -t $SESSION
