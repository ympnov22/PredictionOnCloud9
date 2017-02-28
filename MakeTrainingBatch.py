GENOM_NUM = 30

f = open('Training.sh', 'w')

for i in range(GENOM_NUM):
    f.write("echo training G{}\n".format(i))
    f.write("mv ./G{}/genom.csv ./\n".format(i))
    f.write("python StockPriceTraining.py\n")
    f.write("mv *.csv ./G{}/\n".format(i))
    f.write("mv model.* ./G{}/\n".format(i))
    f.write("mv checkpoint ./G{}/\n".format(i))
    f.write("\n")

f.close() 






