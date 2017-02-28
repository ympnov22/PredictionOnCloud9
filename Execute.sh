@echo off
echo starting batch

echo generate genom
python GenerateGenom.py

echo 1st generation
sh Training.sh
python GeneticAlgorithm.py

echo 2ed generation
sh Training.sh
python GeneticAlgorithm.py

echo 3rd generateion
sh Training.sh
python GeneticAlgorithm.py

echo 4th generation
sh Training.sh
python GeneticAlgorithm.py