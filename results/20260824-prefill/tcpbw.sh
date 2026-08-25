#!/bin/bash
# usage: tcpbw.sh <client_host> <server_ip> <port> <label>
CLI=$1; IP=$2; PORT=$3; LBL=$4
pkill -f "nc -l -p $PORT" 2>/dev/null
nohup timeout 90 nc -l -p $PORT >/dev/null 2>&1 </dev/null &
sleep 2
ssh $CLI "S=\$(date +%s%N); dd if=/dev/zero bs=1M count=2000 2>/dev/null | timeout 80 nc -q1 $IP $PORT; E=\$(date +%s%N); echo \"$LBL \$(( 2000000 / ((E-S)/1000000) )) MB/s\""
pkill -f "nc -l -p $PORT" 2>/dev/null
