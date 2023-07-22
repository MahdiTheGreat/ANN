
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')

def sigmoid(a):
    return 1/(1+np.exp(a))

def sigmoidDerivitive(a):
 return sigmoid(a)*(1-sigmoid(a))

def randIntializer(element):
    return random()

def sech(a):
    return np.divide(1,np.tanh(a))



def neuralWeightMaker(neuralMatrix):
    wgralij = []
    for l in range(0, len(neuralMatrix) - 1):
        wgralij.append(np.zeros((neuralMatrix[l], neuralMatrix[l + 1]), dtype=float))
    return wgralij

def bMaker(neuralMatrix):
    bgralj = []
    for l in range(1, len(neuralMatrix)):
        bgralj.append([0 for i in range(0, neuralMatrix[l])])
    return bgralj








class Network:

  def __init__(self,neuralMatrix):

    self.Wlij =[]
    self.Alj=[]
    self.Zlj=[]
    self.input=[]
    self.labels=[]
    self.blj=[]
    self.Wlij=neuralWeightMaker(neuralMatrix)
    self.Alj=bMaker(neuralMatrix)
    self.Zlj=bMaker(neuralMatrix)
    self.blj=bMaker(neuralMatrix)
    print("test")




  def getWights(self):
      return self.Wlij

  def setWeights(self,Wlij):
      self.Wlij=Wlij

  def getAnswers(self):
      return self.Zlj

  def setAnswers(self,Zlj):
      self.Zlj=Zlj

  def getAxons(self):
      return self.Alj

  def setAxons(self, Alj):
      self.Alj = Alj

  def getB(self):
      return self.blj

  def forwardPropogation(self,input):
      print("test")
      for l in range(0,len(self.Wlij)):
        if l == 0:
         temp1=np.multiply(1/255,input)
         temp1=np.transpose(temp1)
         temp2=self.Wlij[l]
         temp3=np.matmul(temp1,temp2)
         print("test")
         self.Zlj[l]=np.add(temp3,self.blj[l])
         self.Alj[l]=sigmoid(self.Zlj[l])
         print("test")


        else:
         temp1=self.Alj[l-1]
         temp2=self.Wlij[l]
         temp3=np.matmul(self.Alj[l-1], self.Wlij[l])
         print("test")
         self.Zlj[l] = np.add(temp3,self.blj[l])
         self.Alj[l] = sigmoid(self.Zlj[l])
         print("test")

      print("test")

  def deriAlijCalculator(self,output):

      deriAlj=[]
      for l in range(1, len(neuralMatrix)):
          deriAlj.append([0 for i in range(0, neuralMatrix[l])])


      lenTemp=len(self.Alj)-1
      temp1=np.multiply(-1,output)
      print()
      temp1=np.transpose(temp1)
      print()
      temp2=np.add(self.Alj[lenTemp],temp1)
      print()
      deriAlj[lenTemp]=np.multiply(2,temp2)
      print("test")

      for l in range(lenTemp,0,-1):
          temp1=sigmoidDerivitive(self.Zlj[l])
          print("test")
          temp2=np.multiply(deriAlj[l],temp1)
          print("test")
          temp3=np.transpose(self.Wlij[l])
          print()
          deriAlj[l - 1] = np.matmul(temp2,temp3)
          print("test")

      print()
      return deriAlj


  def backwardPropogation(self,deriAlij,wgralij,bgralj,input):

      for l in range(len(self.Wlij)-1,-1,-1):

       if(l==0):temp1=input
       else:temp1=np.transpose(self.Alj[l-1])
       print("test")
       temp2=sigmoidDerivitive(self.Zlj[l])
       print("test")
       temp3=np.multiply(temp2,deriAlij[l])
       print("test")
       temp4=np.matmul(temp1,temp3)
       print()
       wgralij[l]=np.add(wgralij[l],(temp4))
       print()

      for l in range(len(self.blj)-1, -1,-1):
          temp1=np.multiply(sigmoidDerivitive(self.Zlj[l]), deriAlij[l])
          print("test")
          bgralj[l] = np.add(bgralj[l],temp1 )
          print("test")

  def costCalculator(self,output):
      print()
      temp1=np.transpose(output)
      print()
      cost=np.subtract(self.Alj[len(self.Alj)-1],temp1)
      print()
      cost=np.multiply(cost,cost)
      print()
      cost=np.sum(cost)
      print()
      return cost




  def learn(self,wgralij,bgralj,batchSize,learningRate):
      print("test")
      temp1=(learningRate/batchSize)
      for l in range(0,len(neuralMatrix)-1):
       print("test")
       temp2=np.multiply(temp1,wgralij[l])
       print("test")
       self.Wlij[l]=np.subtract(self.Wlij[l],temp2)
       print("test")
       self.blj[l] = np.add(self.blj[l], np.multiply(temp1, bgralj[l]))
      print("test")




train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
#num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
num_of_train_images = 1000
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

#num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
num_of_test_images = 1000
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    test_set.append((image, label))

# Plotting an image
#show_image(train_set[0][0])
#plt.show()

neuralMatrix=[784,16,16,10]
network=Network(neuralMatrix)

print("test")

j=0
trainingSize=100
batchSize=10
learningRate=1
epoch=20

wgralij=neuralWeightMaker(neuralMatrix)

bgralj=bMaker(neuralMatrix)

cost=[]
costBatch=[]
costEpoch=[]
costEpochTemp=0
costBatchTemp=0

accuracy=[]
accuracyBatch=[]
accuracyEpoch=[]
accuracyBatchTemp=0
accuracyEpochTemp=0

print()

for k in range(0,epoch):
 trainset=[]
 print()
 for i in range(0,trainingSize):
     trainset.append(random.choice(train_set))
 print()

 for i in range(0,len(trainset)):
     j += 1
     j = j % batchSize
     if (j == 0):
         costBatch.append(costBatchTemp/batchSize)
         costBatchTemp=0
         costEpochTemp+=costBatch[len(costBatch)-1]
         network.learn(wgralij,bgralj,batchSize,learningRate)
         wgralij = neuralWeightMaker(neuralMatrix)
         bgralj = bMaker(neuralMatrix)
         print("test")

     print("test")
     network.forwardPropogation(trainset[i][0])
     print("test")
     deriij=network.deriAlijCalculator(trainset[i][1])
     print("test")
     network.backwardPropogation(deriij,wgralij,bgralj,trainset[i][0])
     print("test")
     cost.append(network.costCalculator(trainset[i][1]))
     print()
     costBatchTemp += cost[len(cost) - 1]
     print()


 costEpoch.append(costEpochTemp/np.floor(num_of_train_images/batchSize))
 costEpochTemp=0

w=network.getWights()
b=network.getB()
#accuracyRatio=accuracy[0]/(epoch*num_of_test_images)
#print("accuracy is")
#print(accuracyRatio)

#this is for each single cost
t=np.arange(0,len(cost),1)
plt.plot(t,cost)
plt.show()

#this is for a cost after a single batch,which is an average of the single costs
t=np.arange(0,len(costBatch),1)
plt.plot(t,costBatch)
plt.show()

#this is for a cost after a single epoch,which is an average of the batch costs
t=np.arange(0,len(costEpoch),1)
plt.plot(t,costEpoch)
plt.show()

#this is for each single error
t=np.arange(0,len(cost),1)
plt.plot(t,np.subtract(1,np.divide(cost,10)))
plt.show()

#this is for a error after a single batch,which is an average of the single errors
t=np.arange(0,len(costBatch),1)
plt.plot(t,np.subtract(1,np.divide(costBatch,10)))
plt.show()

#this is for a error after a single epoch,which is an average of the batch error
t=np.arange(0,len(costEpoch),1)
plt.plot(t,np.subtract(1,np.divide(costEpoch,10)))
plt.show()


print("test")







