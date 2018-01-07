echo training G0
mkdir G0
python StockPriceTraining.py
mv *.csv ./G0/
mv model.* ./G0/
mv checkpoint ./G0/

echo training G1
mkdir G1
python StockPriceTraining.py
mv *.csv ./G1/
mv model.* ./G1/
mv checkpoint ./G1/

echo training G2
mkdir G2
python StockPriceTraining.py
mv *.csv ./G2/
mv model.* ./G2/
mv checkpoint ./G2/

echo training G3
mkdir G3
python StockPriceTraining.py
mv *.csv ./G3/
mv model.* ./G3/
mv checkpoint ./G3/

echo training G4
mkdir G4
python StockPriceTraining.py
mv *.csv ./G4/
mv model.* ./G4/
mv checkpoint ./G4/

echo training G5
mkdir G5
python StockPriceTraining.py
mv *.csv ./G5/
mv model.* ./G5/
mv checkpoint ./G5/

echo training G6
mkdir G6
python StockPriceTraining.py
mv *.csv ./G6/
mv model.* ./G6/
mv checkpoint ./G6/

echo training G7
mkdir G7
python StockPriceTraining.py
mv *.csv ./G7/
mv model.* ./G7/
mv checkpoint ./G7/

echo training G8
mkdir G8
python StockPriceTraining.py
mv *.csv ./G8/
mv model.* ./G8/
mv checkpoint ./G8/

echo training G9
mkdir G9
python StockPriceTraining.py
mv *.csv ./G9/
mv model.* ./G9/
mv checkpoint ./G9/

