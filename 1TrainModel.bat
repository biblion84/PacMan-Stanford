DEL /F DistanceGlued2.csv

python TrainedModels\GlueTrainingDatas.py
python TrainedModels\DecisionTreeTrain.py

python pacman.py -l smallClassic -p Matrix --frameTime 0 -n 10