GENOM_NUM = 1

f = open('Prediction.sh', 'w')

for i in range(GENOM_NUM):
    f.write("echo Prediction G{}\n".format(i))
    f.write("mv ./G{}/genom.csv ./\n".format(i))
    f.write("mv ./G{}/model.* ./\n".format(i))
    f.write("mv ./G{}/checkpoint ./\n".format(i))
    f.write("python StockPricePrediction.py\n")
    f.write("mv PredictionResult.csv ./G{}/\n".format(i))
    f.write("mv genom.csv ./G{}/\n".format(i))
    f.write("mv model.* ./G{}/\n".format(i))
    f.write("mv checkpoint ./G{}/\n".format(i))
    f.write("\n")

f.close() 






