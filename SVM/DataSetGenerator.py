import joblib
import random

#------------------------------------------------------------------------------
def loadData(path):
    return joblib.load(path)

#------------------------------------------------------------------------------
def createBasicDataArrays(data, labels):
    sizeData = [0 for i in range(41)]
    fullData = [ sizeData.copy() for i in range(len(data))]
    for i in range(0, len(data)):
        for j in range(len(data[i])):
            fullData[i][j] = data[i][j]
        fullData[i][40] = labels[i]

    return fullData

#------------------------------------------------------------------------------
def featAm(data, labels, featAmount):
    newData = []
    for set in range(0, 3):
        sizeData = [0 for i in range(featAmount + 1)]
        fullData = [ sizeData.copy() for i in range(len(data))]
        features = []
        ready = False;
        while ready is False:
            auxFeat =random.randint(0,39)
            if auxFeat not in features:
                features.append(auxFeat)
            if len(features) == featAmount:
                ready = True

        for i in range(len(data)):
            for j in range(featAmount):
                fullData[i][j] = data[i][features[j]]
            fullData[i][featAmount] = labels[i]
        auxData = [set, features, fullData]
        newData.append(auxData)

    return newData

#------------------------------------------------------------------------------
def segmentate(dataSet, mode):
    if mode == 0:
        trainData = []
        testData = []
        unitData = [i for i in range(len(dataSet))]
        trainAmount = int(len(dataSet) * 0.7)
        testAmount = len(dataSet) - trainAmount
        for i in range(0, trainAmount):
            auxPos = random.randint(0, len(unitData) - 1)
            trainData.append(dataSet[auxPos])
            unitData.pop(auxPos)
        for i in range(0, testAmount):
            auxPos = random.randint(0, len(unitData) - 1)
            testData.append(dataSet[auxPos])
            unitData.pop(auxPos)

        generateTxt([0, trainData, testData])

    else:
        for currentData in range(len(dataSet)):
            trainData = []
            testData = []
            realDataSet = dataSet[currentData][2]
            unitData = [i for i in range(len(realDataSet))]
            trainAmount = int(len(realDataSet) * 0.7)
            testAmount = len(realDataSet) - trainAmount
            for i in range(0, trainAmount):
                auxPos = random.randint(0, len(unitData) - 1)
                trainData.append(realDataSet[auxPos])
                unitData.pop(auxPos)
            for i in range(0, testAmount):
                auxPos = random.randint(0, len(unitData) - 1)
                testData.append(realDataSet[auxPos])
                unitData.pop(auxPos)
            generateTxt([dataSet[currentData][0], trainData, testData])

    return True

#------------------------------------------------------------------------------
def generateTxt(data):
    name = str(data[0])
    path = 'Data/'
    trainFile = open(path + str(name) + str(len(data[1][0])) + 'train.txt', 'w')
    for i in range(len(data[1])):
        for j in range(len(data[1][i])):
            trainFile.write(str(data[1][i][j]))
            trainFile.write(' ')
        trainFile.write('\n')
    trainFile.close()

    testFile = open(path +  str(name) + str(len(data[1][0])) + 'test.txt', 'w')
    for i in range(len(data[2])):
        for j in range(len(data[2][i])):
            testFile.write(str(data[2][i][j]))
            testFile.write(' ')
        testFile.write('\n')
    testFile.close

    return True


#------------------------------------------------------------------------------


dataR = loadData('ravdess_speech_data.gz')
labelsR = loadData('ravdess_numeric_labels.gz')
fullDataR = createBasicDataArrays(dataR, labelsR)
newDataRTen = featAm(dataR, labelsR, 10)
newDataRTwe = featAm(dataR, labelsR, 20)
joblib.dump(fullDataR, 'fullData.gz')
joblib.dump(newDataRTen, 'dataTen.gz')
joblib.dump(newDataRTwe, 'dataTwe.gz')
segmentate(fullDataR, 0)
segmentate(newDataRTen, 1)
segmentate(newDataRTwe, 1)

