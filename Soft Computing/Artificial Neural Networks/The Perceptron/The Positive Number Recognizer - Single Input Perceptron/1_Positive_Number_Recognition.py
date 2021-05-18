#Trying to train the computer to recognize positive numbers: The single Perceptron with single input


import sys
import random


def generateData():
    inputFile=open((sys.argv[0]).split(".")[0] + "_TrainingData.txt", "w")
    for i in range(10000):
        nextInt=random.randint(0, 10000)
        multiplier=random.choice([-1, 1])
        if nextInt * multiplier>=0:
            category=1
        else:
            category=-1
        inputFile.writelines(str(nextInt * multiplier) + " " + str(category) + "\n")


def takeInput():
    inputFile=open((sys.argv[0]).split(".")[0] + "_TrainingData.txt", "r")
    line=(inputFile.read()).splitlines()
    trainingData=[]
    for i in range(len(line)):
        tempData=[int(i) for i in line[i].split()]
        trainingData.append(tempData)
    return trainingData


def train(trainingData):
    iterations=0
    ip=trainingData[0][1]
    weight=random.randint(1, 10)
    threshold=1000
    
    while iterations < len(trainingData) * 10:
        randomIndex=random.randint(0, len(trainingData)-1)
        ip=trainingData[randomIndex][1]
        if (trainingData[randomIndex][0] * weight - threshold) >=0:
            op=1
        else:
            op=-1
        iterations+=1
        
        if not op==ip:
            weight+=ip * trainingData[randomIndex][0]
        
        print ("iterations=", iterations, " X=", trainingData[randomIndex][0], " Y=", ip, " Y'=", op, " W=", weight, sep="")
    
    print ("Network Trained with weight set to", weight)
    return weight, threshold


def classify(inputInt, weight, threshold):
    if (inputInt * weight - threshold) >=0:
        print ("POSITIVE")
    else:
        print ("NEGATIVE")


def test(weight, threshold):
    while(1):
        inputInt=int(input("Please enter any integer: "))
        classify(inputInt, weight, threshold)


def main():

    generateData()

    trainingData=takeInput()
    
    weight, threshold=train(trainingData)
    
    test(weight, threshold)


if __name__=="__main__":
	main()


