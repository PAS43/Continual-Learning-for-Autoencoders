import keras.datasets.mnist as mnist
from matplotlib.pyplot import imshow
import keras.datasets.cifar10 as cifar
import numpy as np

np.random.seed(5337)


def getDatasets():
    dataset = {
        'mnist': mnist.load_data(),
        'caifar': cifar.load_data(),
    }
    return dataset

class swapMatrix:
    def genRandomVec(self, imageSize, probabilityOfPixelSwap):
        pixelStay = 1.0 - probabilityOfPixelSwap
        pixelSwap = []
        for row in range(imageSize):
            pixelSwap.append(np.random.choice(2, imageSize, p=[pixelStay, probabilityOfPixelSwap]))
        return pixelSwap

    def genRandomLocation(self, imageSize, randomVec):
        swapMatrix = []
        totalPixels = imageSize * imageSize
        for row in randomVec:
            for element in row:
                if element == 1:
                    swapMatrix.append([np.random.randint(0, totalPixels), np.random.randint(0, totalPixels)])
        return swapMatrix

    def getSwapMatrix(self, imageSize, probabilityOfPixelSwap):
        randomDim = self.genRandomVec(imageSize, 0.2)
        swapMatrix = self.genRandomLocation(imageSize, randomDim)
        return swapMatrix

    def swapMnist(self, dataset, swapMatrix, size):
        alterdImage = []
        for x in swapMatrix:
            eachImage = dataset.flatten()
            eachImage[x[0]] = eachImage[x[1]]
            alterdImage.append(np.reshape(eachImage, (size, size)))
        return alterdImage

    def makeNewMnistDataset(self, dataset, probabilityOfSwap):
        swapVec = self.getSwapMatrix(28, probabilityOfSwap)
        test = []
        for row in dataset:
            test.append(self.swapMnist(row, swapVec, 28))
        return test

    def swapCaifar(self, dataset, swapMatrix, size):
        alterdImage = []
        for x in swapMatrix:
            eachImage = dataset.reshape(32*32,3)
            eachImage[x[0]] = eachImage[x[1]]
            alterdImage.append(eachImage.reshape(32,32,3))
        return alterdImage

    def makeNewCaifarDataset(self, dataset, probabilityOfSwap):
        swapVec = self.getSwapMatrix(32, probabilityOfSwap)
        test = []
        for row in dataset:
            test.append(self.swapCaifar(row, swapVec, 32))
        return test

if __name__ == "__main__":

    datasets = getDatasets()
    # mnistSwapMatrix = swapMatrix().getSwapMatrix(28, 0.2)
    # cifarSwapMatrix = swapMatrix().getSwapMatrix(32, 0.2)

    mnist = datasets['mnist']
    cifar = datasets['caifar']

    numberOfInstances = 1000

    m2 = swapMatrix().makeNewMnistDataset(mnist[0][0][0:numberOfInstances], 0.2)
    c2 = swapMatrix().makeNewCaifarDataset(cifar[0][0][0:numberOfInstances], 0.2)
    m4 = swapMatrix().makeNewMnistDataset(mnist[0][0][0:numberOfInstances], 0.4)
    c4 = swapMatrix().makeNewCaifarDataset(cifar[0][0][0:numberOfInstances], 0.4)
    m6 = swapMatrix().makeNewMnistDataset(mnist[0][0][0:numberOfInstances], 0.6)
    c6 = swapMatrix().makeNewCaifarDataset(cifar[0][0][0:numberOfInstances], 0.6)

    np.save("Mnist20%", m2, allow_pickle=True)
    np.save("Mnist40%", m4, allow_pickle=True)
    np.save("Mnist60%", m6, allow_pickle=True)

    np.save("Cifar20%", c2, allow_pickle=True)
    np.save("Cifar40%", c4, allow_pickle=True)
    np.save("Cifar60%", c6, allow_pickle=True)



    #Write swap function!!
    print("STOP!")