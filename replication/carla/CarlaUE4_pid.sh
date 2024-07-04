#!/bin/sh

if [ $# -eq 0 ]; then
    pid_file=$(mktemp ../carla_server_pid_XXXXXX.txt)
else
    pid_file="$1"
    # removes the argument
    shift
fi

UE4_TRUE_SCRIPT_NAME=$(echo \"$0\" | xargs readlink -f)
UE4_PROJECT_ROOT=$(dirname "$UE4_TRUE_SCRIPT_NAME")
chmod +x "$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
"$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 "$@" & echo $! > "$pid_file"