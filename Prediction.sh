echo Prediction G0
mv ./G0/genom.csv ./
mv ./G0/model.* ./
mv ./G0/checkpoint ./
python StockPricePrediction.py
mv PredictionResult.csv ./G0/
mv genom.csv ./G0/
mv model.* ./G0/
mv checkpoint ./G0/

