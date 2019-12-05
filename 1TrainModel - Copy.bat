DEL /F DistanceGluedWithout2.csv

python TrainedModels\GlueTrainingDatasWithout.py
python TrainedModels\DecisionTreeTrainWithout.py

python pacman.py -l smallClassic -p Matrix --frameTime 0 -n 10