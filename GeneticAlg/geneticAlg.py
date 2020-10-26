import joblib
import random
import math

#------------------------------------------------------------------------------
VAL_FOR_MIN = 9999
VAL_FOR_MAX = -9999

#------------------------------------------------------------------------------
def geneticAlg( ranks, popAmount, featureAmount, classes, modGen, modMut ):
    ravdess_data = joblib.load( "ravdess_speech_data.gz" )
    ravdess_labels = joblib.load( "ravdess_numeric_labels.gz" )

    limits = getLimits( ravdess_data, ravdess_labels, featureAmount )
    samples = codify( ravdess_data, ravdess_labels, limits, ranks, featureAmount )

    population = initPopulation( popAmount, featureAmount, ranks, classes, featureAmount )

    popResults = evalPop( population, samples, classes, featureAmount )

    finalRule = startGenerations( modGen, modMut, population, samples, classes, featureAmount, popResults, ranks )

    #finalResult = evalRes( finalRule, samples, classes, featureAmount )
    #print( finalResult )

    return finalRule

#------------------------------------------------------------------------------
def startGenerations( modNext, modGen, population, samples, classes, featureAmount, popResults, ranks ):
    if len( population ) == 1:
        return population
    else:
        population = nextGeneration( modNext, popResults, population )
        population = modNextGeneration( modGen, population, featureAmount, ranks )
        popResults = evalPop( population, samples, classes, featureAmount )
        print( popResults )
        return startGenerations( modNext, modGen, population, samples, classes, featureAmount, popResults, ranks )


#------------------------------------------------------------------------------
def getLimits( data, labels, features ):
    limits = [ [ VAL_FOR_MIN, VAL_FOR_MAX ] for i in range( features ) ]
    for i in range( 0, len( data ) ):
        for j in range( 0, features ):
            if data[ i ][ j ] < limits[ j ][ 0 ]:
                limits[ j ][ 0 ] = data[ i ][ j ]
            if data[ i ][ j ] > limits[ j ][ 1 ]:
                limits[ j ][ 1 ] = data[ i ][ j ]

    for i in range( 0, len( limits ) ):
        limits[ i ][ 1 ] += abs( limits[ i ][ 0 ] )

    return limits

#------------------------------------------------------------------------------
def codify( oldData, labels, limits, ranks, features ):
    subData = [ 0 for i in range( features + 1 ) ]
    data = [ subData.copy() for i in range( len( oldData ) ) ]
    for i in range( 0, len( oldData ) ):
        for j in range( 0, features ):
            data[ i ][ j ] = getCode( oldData[ i ][ j ], limits[ j ], ranks )
        data[ i ][ features ] = labels[ i ]

    return data

#------------------------------------------------------------------------------
def getCode( val, limit, ranks ):
    val += abs( limit[ 0 ] )
    boundary = limit[ 1 ] / ( ranks - 1 )
    intCode = math.floor( val / boundary )
    lenCode = math.log( ranks, 2 )

    code = str( "{0:b}".format( intCode ) )
    for i in range( 0, int( lenCode ) - len( code ) ):
        code += "0"

    return code

#------------------------------------------------------------------------------
def initPopulation( popAmount, featureAmount, ranks, classes, features ):
    popfeatures = [ "" for i in range( featureAmount + 1 ) ]
    individuals = [ popfeatures.copy() for i in range( classes * 60 ) ]

    population = [ "" for i in range( popAmount ) ]

    for i in range( 0, popAmount ):
        for j in range( 0, len( individuals ) ):
            for k in range( 0, featureAmount ):
                popfeatures[ k ] = randCode( ranks )
            individuals[ j ] = popfeatures.copy()
            popClass = j % classes
            individuals[ j ][ features ] = str( popClass )
        population[ i ] = individuals.copy()

    return population

#------------------------------------------------------------------------------
def evalPop( population, samples, classes, features ):
    popResults = [ [ 0, 0 ] for i in range( len( population ) ) ]
    for i in range( 0, len( population ) ):
        popResults[ i ][ 0 ] = i
        popResults[ i ][ 1 ] = solResults( population[ i ], samples, classes, features )

    return popResults

#------------------------------------------------------------------------------
def evalRes( population, samples, classes, features ):
    popResults = [ [ 0, 0 ] for i in range( len( population ) ) ]
    for i in range( 0, len( population ) ):
        popResults[ i ][ 0 ] = i
        popResults[ i ][ 1 ] = solRule( population[ i ], samples, classes, features )

    return popResults

#------------------------------------------------------------------------------
def solRule( solution, samples, classes, features ):
    confRow = [ 0 for i in range( classes ) ]
    confMatrix = [ confRow.copy() for i in range( classes ) ]
    testSampleSize = math.floor( len( samples ) * 0.7 )

    goodCount = 0
    badCount = 0
    for i in range( testSampleSize + 1, len( samples ) ):
        badCount += 1
        goodCount = checkSample( samples[ i ], solution, confMatrix, goodCount, features )

    if goodCount != 0:
        fitness = goodCount / badCount
    else:
        fitness = 0

    return fitness

#------------------------------------------------------------------------------
def solResults( solution, samples, classes, features ):
    confRow = [ 0 for i in range( classes ) ]
    confMatrix = [ confRow.copy() for i in range( classes ) ]
    testSampleSize = math.floor( len( samples ) * 0.7 )

    goodCount = 0
    badCount = 0
    for i in range( 0, testSampleSize ):
        badCount += 1
        goodCount = checkSample( samples[ i ], solution, confMatrix, goodCount, features )

    if goodCount != 0:
        fitness = goodCount / badCount
    else:
        fitness = 0

    return fitness

#------------------------------------------------------------------------------
def checkSample( sample, solution, confMatrix, goodCount, features ):
    breakLoop = False
    for i in range( 0, len( solution ) ):
        validate = True
        for j in range( 0, features ):
            if sample[ j ] != solution[ i ][ j ]:
                validate = False
            if j == features - 1 and validate is True:
                if str( sample[ features ] ) == solution[ i ][ features ]:
                    goodCount += 1
                    breakLoop = True
        if breakLoop:
            break

    return goodCount

# ------------------------------------------------------------------------------

def randCode( ranks ):
    lenCode = math.log(ranks, 2)
    code = ""

    for i in range(0, int(lenCode)):
        randBit = random.randint(0, 1)
        code += str(randBit)

    return code

#------------------------------------------------------------------------------
def nextGeneration( mod, popResults, population ):
    if mod == 0 :
        return genElite( popResults, population )
    if mod == 1:
        return genRoulette( popResults, population )
    if mod == 2:
        return genTournament( popResults, population )

#------------------------------------------------------------------------------
def genTournament( popResults, population ):
    newPopSize = math.floor( len( population ) / 2 )
    newPopulation = [population[0].copy() for i in range(newPopSize)]
    for i in range( 0, newPopSize ):
        if popResults[ i ][ 1 ] > popResults[ i + ( math.floor( len( population ) / 2 ) ) ][ 1 ]:
            newPopulation[ i ] = population[ popResults[ i ][ 0 ] ]
        else:
            newPopulation[ i ] = population[ popResults[ i + ( math.floor( len( population ) / 2 ) ) ][ 0 ] ]

    return newPopulation

#------------------------------------------------------------------------------
def genElite( popResults, population ):
    newPopSize = math.floor( len( population ) / 2 )
    newPopulation = [ population[ 0 ].copy() for i in range( newPopSize ) ]
    for i in range( 0, newPopSize ):
        actualMin = [ 0, VAL_FOR_MIN ]
        for j in range( 0, len( popResults ) ):
            if popResults[ j ][ 1 ] < actualMin[ 1 ]:
                actualMin = popResults[ j ]
                actualMin[ 0 ] = j
        popResults.pop( actualMin[ 0 ] )

    for i in range( 0, newPopSize ):
        newPopulation[ i ] = population[ popResults[ i ][ 0 ] ]

    return newPopulation

#------------------------------------------------------------------------------
def genRoulette( popResults, population ):
    newPopSize = math.floor( len( population ) / 2 )
    newPopulation = [ population[ 0 ].copy() for i in range( newPopSize ) ]
    countTotal = 0
    for i in range( 0, len( popResults ) ):
        countTotal += popResults[ i ][ 1 ]
    for i in range( 0, newPopSize ):
        actualMin = [ 0, VAL_FOR_MIN ]
        for j in range( 0, len( popResults ) ):
            newPopResult = popResults[ j ][ 1 ] / countTotal
            if newPopResult < actualMin[ 1 ]:
                actualMin = popResults[ j ]
                actualMin[ 0 ] = j
        popResults.pop( actualMin[ 0 ] )

    for i in range( 0, newPopSize ):
        newPopulation[ i ] = population[ popResults[ i ][ 0 ] ]

    return newPopulation

#------------------------------------------------------------------------------
def modNextGeneration( mod, population, features, ranks ):
    if mod == 0:
        return mutation( population, features, ranks )
    elif mod == 1:
        return crossing( population )
    elif mod == 2:
        if random.randint( 0, 1) == 0:
            population = mutation( population, features, ranks)
            population = crossing( population )
        else:
            population = crossing( population )
            population = mutation( population )

        return population

#------------------------------------------------------------------------------
def mutation( population, features, ranks ):
    for i in range( 0, len( population ) ):
        randRules = random.randint( 0, len( population[ i ] ) / 4 )
        for j in range( 0, randRules ):
            randRule = random.randint( 0, len( population[ i ] ) - 1 )
            randMuts = random.randint(0, int( features / 2 ) )
            for k in range( 0, randMuts ):
                mutPos = random.randint(0, features)
                population[ i ][ randRule ][ mutPos ] = randCode( ranks )
    return population

#------------------------------------------------------------------------------
def crossing( population ):
    for i in range( 0, len( population ) ):
        randMember = random.randint( 0, len( population ) - 1 )
        randRules = random.randint( 0, len( population[ i ] ) - 1 )
        for j in range( 0, randRules ):
            randRule = random.randint( 0, len( population[ i ] ) - 1)
            tempRule = population[ i ][ randRule ]
            population[ i ][ randRule ] = population[ randMember ][ randRule ]
            population[ randMember ][ randRule ] = tempRule

    return population

#------------------------------------------------------------------------------


#Ranks, Population amount, feature amount, classes, modGen, modMut
geneticAlg( 4, 100, 5, 8, 2, 0 )

