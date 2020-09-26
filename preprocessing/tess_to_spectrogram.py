import time
import librosa.display
import matplotlib.pyplot as plt
from os import listdir

#------------------------------------------------------------------------------
def main():
    searchDir()

#------------------------------------------------------------------------------
def searchDir():
    # myPath = "/home/david/"
    # myPath += "Documentos/Universidad/9no Semestre/Sistemas Inteligentes/Proyecto/TESS Toronto emotional speech set" \
    #           " data/"
    myPath = './TESS/'

    for fDir in listdir( myPath ):
        if fDir=='.DS_Store':
            pass
        else:
            strEmo = getStrEmo( fDir )
            audioDir = myPath + fDir
            for fA in listdir( audioDir ):

                audioPath = audioDir + '/' + fA
                specto = getSpectrogrm( audioPath )
                saveImg( specto, strEmo, fA )



#------------------------------------------------------------------------------
def getSpectrogrm( audioPath ):

    x, sr = librosa.load( audioPath )
    X = librosa.stft( x )
    Xdb = librosa.amplitude_to_db( abs( X ) )
    plt.figure( figsize=( 20, 5 ) )
    librosa.display.specshow( Xdb, sr = sr, x_axis="time", y_axis="hz" )
    plt.colorbar()
    #plt.show()
    #plt.savefig(  )
    time.sleep( 1 )

    return plt

#------------------------------------------------------------------------------
def saveImg( specto, emotion, nameSpecto ):
    nameSpecto = nameSpecto[:-1]
    nameSpecto = nameSpecto[:-1]
    nameSpecto = nameSpecto[:-1]
    print(nameSpecto)
    nameSpecto += "png"
    newName = emotion + "_" + nameSpecto
    savePath = "./models/spectograms/" + newName
    specto.savefig( savePath )

#------------------------------------------------------------------------------
def getStrEmo( name ):
    if "angry" in name:
        return "angry"
    elif "disgust" in name:
        return "disgust"
    elif "Fear" in name or "fear" in name:
        return "fear"
    elif "happy" in name:
        return "happy"
    elif "neutral" in name:
        return "neutral"
    elif "surprise" in name:
        return "surprise"
    elif "sad" in name or "Sad" in name:
        return "sad"

#------------------------------------------------------------------------------

main()