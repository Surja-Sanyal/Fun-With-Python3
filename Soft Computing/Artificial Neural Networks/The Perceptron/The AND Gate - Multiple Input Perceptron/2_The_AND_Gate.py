#The AND Gate: The single Perceptron with variable inputs


import sys
import random


def getTrainingDataFile():
    return (sys.argv[0]).split(".")[0] + "_TrainingData.txt"


def product(args):
    if len(args) > 1:
        return args[0] * product(args[1:])
    else:
        return args[0]


def generateData():
    inputFile=open(getTrainingDataFile(), "w")
    for i in range(10000):
        ip=[]
        for j in range(random.randint(2, 10)):
            inputValue=round(random.uniform(0.0, 1.0), 1)
            ip.append(inputValue)
            inputFile.writelines(str(inputValue)+" ")
        
        if round(product([round(x) for x in ip[1:]]), 1) >= ip[0]:
            category=1
        else:
            category=-1
        inputFile.writelines(str(category) + "\n")


def takeInput():
    inputFile=open(getTrainingDataFile(), "r")
    line=(inputFile.read()).splitlines()
    trainingData=[]
    for i in range(len(line)):
        tempData=[float(i) for i in line[i].split()]
        trainingData.append(tempData)
    return trainingData


def train(trainingData):
    iterations=0
    
    weight=[-1]
    for i in range(max(len(x) for x in trainingData)-1):
        weight.append(random.randint(1, 5))

    
    while iterations < len(trainingData) * 10:
        randomIndex=random.randint(0, len(trainingData)-1)
        category=trainingData[randomIndex][-1]
        if (product([x * y for x, y in zip(trainingData[randomIndex][1:], weight[1:])]) + trainingData[randomIndex][0] * weight[0] >=0.5):
            classification=1
        else:
            classification=-1
        iterations+=1
        
        if not classification==category:
            for i in range(1, len(trainingData[randomIndex])):
                weight[i]+=round(category * trainingData[randomIndex][i], 1)
                weight[i]=round(weight[i], 1)
        
        print ("iterations=", iterations, " X=", trainingData[randomIndex][0:-1], " Y=", category, " Y'=", classification, " W=", weight, sep="")
    
    print ("Network Trained with weight set to", weight)
    return weight


def classify(inputs, weight):
    if (product([x * y for x, y in zip(inputs[1:], weight[1:])]) + inputs[0] * weight[0] >=0.5):
        print ("POSITIVE")
    else:
        print ("NEGATIVE")


def test(weight):
    while(1):
        inputs=[float(x) for x in list((input("Please enter space separated numbers: ")).split())]
        classify(inputs, weight)


def main():

    generateData()

    trainingData=takeInput()
    
    weight=train(trainingData)
    
    test(weight)


if __name__=="__main__":
	main()


