DEL /F DistanceReflexMatrixbatchWithoutGhost0.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost1.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost2.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost3.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost4.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost5.csv
DEL /F DistanceReflexMatrixbatchWithoutGhost6.csv

start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost0.csv" 
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost1.csv" 
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost2.csv" 
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost3.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost4.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost5.csv" -g DirectionalGhost
start "" pypy pacman.py -l smallClassicWithout -p ReflexAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithoutGhost6.csv" -g DirectionalGhost