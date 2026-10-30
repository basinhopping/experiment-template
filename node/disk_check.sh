#!/usr/bin/env bash
# Is there room for what a step will add? Prints the free space of the home folder's file system and exits 1 when
# it is below the gigabytes given (count what the step writes: an archive, a run's folders, an environment):
#     bash ~/<project>/node/disk_check.sh 25
set -euo pipefail
NEED="${1:?usage: disk_check.sh <gigabytes the next step needs>}"
FREE_KB=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
echo "free: ${FREE_GB} GB in $(df -P "$HOME" | awk 'NR == 2 {print $6}'); needed: $NEED GB"
if [ "$FREE_GB" -lt "$NEED" ]; then echo "NOT ENOUGH DISK: stop here and paste this"; exit 1; fi
