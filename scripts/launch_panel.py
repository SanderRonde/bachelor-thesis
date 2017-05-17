import os

def main():
    os.system('tmux new-session -d \'cd scripts/scripts/; echo "loading source..."; source activate nn && bash -i\'')
    os.system('tmux split-window -h -p \'33\' \'htop\'')
    os.system('tmux split-window -t 1 -v \'watch -n 1 -d nvidia-smi\'')
    os.system('tmux select-pane -L')
    os.system('tmux new-window \'main panel\'')
    os.system('tmux -2 attach-session -d')

if __name__ == '__main__':
    main()
