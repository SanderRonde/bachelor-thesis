#!/usr/bin/env bash

panes = $((tmux list-panes -a))
echo "$panes"