DEL /F DistanceReflexMatrixbatch0.csv
DEL /F DistanceReflexMatrixbatch1.csv
DEL /F DistanceReflexMatrixbatch2.csv
DEL /F DistanceReflexMatrixbatch3.csv
DEL /F DistanceReflexMatrixbatch4.csv
DEL /F DistanceReflexMatrixbatch5.csv
DEL /F DistanceReflexMatrixbatch6.csv

start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch0.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch1.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch2.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch3.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch4.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch5.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassic -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatch6.csv" -g DirectionalGhost