DEL /F VoteBatch0.csv
DEL /F VoteBatch1.csv
DEL /F VoteBatch2.csv
DEL /F VoteBatch3.csv
DEL /F VoteBatch4.csv
DEL /F VoteBatch5.csv
DEL /F VoteBatch6.csv

start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch0.csv" 
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch1.csv" 
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch2.csv" 
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch3.csv" 
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch4.csv" -g DirectionalGhost
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch5.csv" -g DirectionalGhost
start "" pypy pacman.py  -p VoteAgent -q -n 50 -a filesave="VoteBatch6.csv" -g DirectionalGhost