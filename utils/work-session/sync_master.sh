#!/bin/bash

#~ USAGE: ./$0 remote|local
#~ SYNOPSIS: sync master

# set:
WDIR="/home/verde/Desktop/pro-QUAD4M/gitlab" # directory containing "pro-quad4m" folder
WFLD="pro-quad4m-master" # name for the synchronized folder

# exit when any command fails
set -e

if [ "$1" == "remote" ] || [ "$1" == "local" ] ; then
    
    # create $WFLD if non-existent
    mkdir -p "$WDIR/$WFLD"
    
    # echo $WFLD
    echo "---"
    echo "DIR-SYNC: $WDIR/$WFLD"
    echo "---"
    
    # make a copy to "old_remote|old_local" folder
    rsync -a $WDIR/$WFLD $WDIR/old_$1
fi

if [ "$1" == "remote" ] ; then
        
    # remove old zip
    rm -f "$WDIR/$WFLD.zip"

    # download zip
    wget "https://gitlab.rm.ingv.it/rodolfo.puglia/pro-quad4m/-/archive/master/$WFLD.zip" -O "$WDIR/$WFLD.zip"

    # extract new master
    unzip -o "$WDIR/$WFLD.zip" -d "$WDIR"
fi

if [ "$1" == "local" ] ; then

    # cd to make rsync exclusion effective
    cd "$WDIR/$WFLD"

    # sync master
    OPTIONS="-av --exclude=\".git\""
    rsync $OPTIONS $WDIR/pro-quad4m/* $WDIR/$WFLD
fi
