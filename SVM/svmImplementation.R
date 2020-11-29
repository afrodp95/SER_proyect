# Attach Packages
library(rlang)    # data manipulation and visualization
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology
library(ISLR)         # contains example data set "Khan"
library(RColorBrewer) # customized coloring of plots

basic_path <- "/home/david/Documentos/Universidad/9no Semestre/Sistemas Inteligentes/Proyecto/SVM/Data/"
file_list <- list.files(basic_path)
file_list

expVectorTr = c()
accVectorTr = c()
expVectorTs = c()
accVectorTs = c()

krStr = c("Radial", "Linear", "Sigmoid", "Polynomial" )

for(i in seq_along(file_list))
{
  if( i %% 2 == 0)
  {
    #Path to the data set
    testPath = paste(basic_path, file_list[i-1], sep="")
    trainPath = paste(basic_path, file_list[i], sep="")
    
    #Load the data set
    trainTab = read.table(trainPath)
    testTab = read.table(testPath)
    
    #Number of samples per data set
    trRow = NROW(trainTab)
    tsRow = NROW(testTab)
    
    #Number of classes. Possible values: 4 for 4 & 1 for 10
    sub = substr(file_list[i], 1, 1)
    
    #Classes
    classes = c(0, 1, 2, 3, 4, 5, 6, 7)
      
    #Number of features
    val = length(trainTab)
      
    #Setting the dataset
    trainTab[,val]= factor(trainTab[,val], levels = classes)
    testTab[,val]= factor(testTab[,val], levels = classes)
      
    #Radial kernel
    classifierRad = svm(formula = trainTab[,val] ~ ., data = trainTab, kernel = 'radial', scale=FALSE)
      
    #Linear kernel
    classifierLin = svm(formula = trainTab[,val] ~ ., data = trainTab, kernel = 'linear', scale=FALSE)
      
    #Sigmoid kernel
    classifierSig = svm(formula = trainTab[,val] ~ ., data = trainTab, kernel = 'sigmoid', scale=FALSE)
      
    #Polynomial kernel
    classifierPol = svm(formula = trainTab[,val] ~ ., data = trainTab, kernel = 'polynomial', scale=FALSE)
    #summary(classifier)
      
    #Training data
    #Radial kernel
    y_predTrRad = predict(classifierRad, newdata = trainTab)
    cmTrRad = table(trainTab[,val], y_predTrRad)
    trRadAcc = sum(diag(cmTrRad))/trRow
    expName = paste(krStr[1], file_list[i])
    expVectorTr = c(expVectorTr, expName)
    accVectorTr = c(accVectorTr, trRadAcc)
      
    #Linear kernel
    y_predTrLin = predict(classifierLin, newdata = trainTab)
    cmTrLin = table(trainTab[,val], y_predTrLin)
    trLinAcc = sum(diag(cmTrLin))/trRow
    expName = paste(krStr[2], file_list[i])
    expVectorTr = c(expVectorTr, expName)
    accVectorTr = c(accVectorTr, trLinAcc)
      
    #Sigmoid Kernel
    y_predTrSig = predict(classifierSig, newdata = trainTab)
    cmTrSig = table(trainTab[,val], y_predTrSig)
    trSigAcc = sum(diag(cmTrSig))/trRow
    expName = paste(krStr[3], file_list[i])
    expVectorTr = c(expVectorTr, expName)
    accVectorTr = c(accVectorTr, trSigAcc)
    
    #Polynomial kernel
    y_predTrPol = predict(classifierPol, newdata = trainTab)
    cmTrPol = table(trainTab[,val], y_predTrPol)
    trPolAcc = sum(diag(cmTrPol))/trRow
    expName = paste(krStr[4], file_list[i])
    expVectorTr = c(expVectorTr, expName)
    accVectorTr = c(accVectorTr, trPolAcc)
      
      
    #Testing data
    
    #Radial kernel
    y_predTsRad = predict(classifierRad, newdata = testTab)
    cmTsRad = table(testTab[,val], y_predTsRad)
    tsRadAcc = sum(diag(cmTsRad))/tsRow
    expName = paste(krStr[1], file_list[i-1])
    expVectorTs = c(expVectorTs, expName)
    accVectorTs = c(accVectorTs, tsRadAcc)
    
    #Linear kernel
    y_predTsLin = predict(classifierLin, newdata = testTab)
    cmTsLin = table(testTab[,val], y_predTsLin)
    tsLinAcc = sum(diag(cmTsLin))/tsRow
    expName = paste(krStr[2], file_list[i-1])
    expVectorTs = c(expVectorTs, expName)
    accVectorTs = c(accVectorTs, tsLinAcc)
      
    #Sigmoid kernel
    y_predTsSig = predict(classifierSig, newdata = testTab)
    cmTsSig = table(testTab[,val], y_predTsSig)
    tsSigAcc = sum(diag(cmTsSig))/tsRow
    expName = paste(krStr[3], file_list[i-1])
    expVectorTs = c(expVectorTs, expName)
    accVectorTs = c(accVectorTs, tsSigAcc)
    
    #Polynomial kernel
    y_predTsPol = predict(classifierPol, newdata = testTab)
    cmTsPol = table(testTab[,val], y_predTsPol)
    tsPolAcc = sum(diag(cmTsPol))/tsRow
    expName = paste(krStr[4], file_list[i-1])
    expVectorTs = c(expVectorTs, expName)
    accVectorTs = c(accVectorTs, tsPolAcc)
    
    #Print cm
    print(cmTrRad)
    print(trRadAcc)
    print(cmTsRad)
    print(tsRadAcc)
    print(cmTrLin)
    print(trLinAcc)
    print(cmTsLin)
    print(tsLinAcc)
    print(cmTrSig)
    print(trSigAcc)
    print(cmTsSig)
    print(tsSigAcc)
    print(cmTrPol)
    print(trPolAcc)
    print(cmTsPol)
    print(tsPolAcc)
  }
}

resultsTrainig = data.frame(expVectorTr, accVectorTr)
resultsTesting = data.frame(expVectorTs, accVectorTs)

write.csv(resultsTrainig, "/home/david/Documentos/Universidad/9no Semestre/Sistemas Inteligentes/Proyecto/SVM/Results/ResultsTrainingR")
write.csv(resultsTesting, "/home/david/Documentos/Universidad/9no Semestre/Sistemas Inteligentes/Proyecto/SVM/Results/ResultsTestingR")
