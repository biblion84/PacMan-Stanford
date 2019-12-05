DEL /F DistanceReflexMatrixbatchWithout0.csv
DEL /F DistanceReflexMatrixbatchWithout1.csv
DEL /F DistanceReflexMatrixbatchWithout2.csv
DEL /F DistanceReflexMatrixbatchWithout3.csv
DEL /F DistanceReflexMatrixbatchWithout4.csv
DEL /F DistanceReflexMatrixbatchWithout5.csv
DEL /F DistanceReflexMatrixbatchWithout6.csv

start "" pypy pacman.py -l smallClassicWithout -p ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout0.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout1.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout2.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout3.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout4.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout5.csv"  
start "" pypy pacman.py -l smallClassicWithout -p  ExpectimaxAgent -q -n 20 -a filesave="DistanceReflexMatrixbatchWithout6.csv"  