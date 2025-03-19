#!/bin/bash

#~ USAGE: ./$0 [pause]
#~ SYNOPSIS: orchestrate pro-QUAD4M execution (in pratice, it executes steps #4, #5 and #6 reported in README.md)
#~ NOTES:
#~  - "WDIR" and "MJSN" parameters below have to be set
#~  - QUAD4M.EXE working directory "$FLD" is the folder read in field 'job_folder' of JSON-file of the input model "$MJSN" 
#~  - software "expect" is required (see https://linux.die.net/man/1/expect) - e.g. "# apt-get install expect"

# set "WDIR" and "MJSN"
WDIR="/home/verde/QUAD4M/pro-quad4m-master"
MJSN="model.json"

# exit when any command fails
set -e

# move to working directory
cd "$WDIR"

# step #4 (execute pre_QUAD4M.py)
python pre_QUAD4M.py "$MJSN" time-history.xacc time-history.yacc

# pause function
function pause() {
    read -p "$*"
}

# pause if any string (e.g. "pause") is passed as "$1"
if [ "$1" ] ; then pause 'Press [Enter] key to continue...' ; fi

# read field 'job_folder' of JSON-file
FLD=$(grep "job_folder" "$MJSN" | cut -d ':' -f 2 | cut -d '"' -f 2)

# move to QUAD4M.EXE working directory and copy here the executable
cd "var/$FLD"
cp ../test_0/QUAD4M.EXE .

# step #5 (execute QUAD4M.EXE through the software "expect")
expect -c '
set timeout -1
set stty_init raw

proc sendline {line} { send -- "$line\n" }

spawn -nottyinit wine QUAD4M.EXE

expect "Enter Input File Name:"
sendline "MDL.Q4I"
expect "Enter Soil Data File Name:"
sendline "SG.DAT"
expect "Enter Output directory"
sendline "\.\\"
expect "Output File Name (without directory name):"
sendline "MDL.Q4O"

expect eof'

# move back to working directory
cd ..
cd ..

# step #6 (execute post_QUAD4M.py)
expect -c '
set timeout -1
set stty_init raw

proc sendline {line} { send -- "$line\n" }

spawn -nottyinit python post_QUAD4M.py "'$MJSN'" var/'$FLD'/MDL.Q4O var/'$FLD'/SG.Q4A var/'$FLD'/borders.txt

expect "Overwrite"
sendline "y"

expect eof' 2> /dev/null
