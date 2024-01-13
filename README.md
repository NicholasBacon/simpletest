```
mkdir files
lalloc 9 -q pdebug lrun  -M -gpu -N 9 -T 1 -g 1 ping_pong >files/mixed
python python.py
```
