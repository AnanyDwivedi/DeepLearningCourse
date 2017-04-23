from keras.layers import Input, Dense
from keras.models import Model
import matplotlib as mpl
mpl.use('Agg')
from keras.datasets import mnist
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def normalizedata(X):
	for i in range(X.shape[1]):
#		print (np.max(np.abs(X[:,i])))
		X[:,i] /= (np.max(np.abs(X[:,i])))
#	print "\n\n"
	return X

def misclass(t,p):
	z=0
	o=0
#	t = numpy.argmax(tru,1)
#	p = numpy.argmax(pred,1)
	
	Zc = 0
	Oc = 0
	for i in range(t.shape[0]):	
		if t[i] == 0:
			Zc+=1
		else:
			Oc+=1
		
		if t[i]!=p[i]:
			if t[i] == 0:
				z+=1
			else:
				o+=1

	print "\n Z = %d O = %d Zc =%d Oc = %d"%(z,o,Zc,Oc)



def data_split(X,y,n_splits=3, test_siz=0.2, random_state=0):
	sss = StratifiedShuffleSplit(n_splits=3, test_size=test_siz, random_state=0)
	sss.get_n_splits(X, y)
	for train_index, test_index in sss.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

	return X_train, X_test, y_train, y_test

def acc(tru,pred):
	if tru.shape[1] > 1:
		tru = np.argmax(tru,1)
		pred = np.argmax(pred,1)
	acc = np.mean(tru == pred)
	return acc


def pred_anomally(x,xd,y_test):
	err = np.mean((x - xd)**2,1)
	pred = np.zeros((y_test.shape[0],1))
	print err
	print err.shape 
	for i in range(y_test.shape[0]):
		if err[i] > 7.5e-5:
			pred[i] = 1
		else:
			pred[i] = 0
	
	accu = acc(y_test,pred)
	return accu,pred

def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 100.0
        lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
        return lrate



# this is the size of our encoded representations
encoding_dim = 16  # 16 floats -> compression of factor 4, assuming the input is 64 floats

# this is our input placeholder
input_size = Input(shape=(64,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='tanh')(input_size)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_size, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_size, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')



#----------------------------------------------------------------------
seed = 7
X = np.load('../Data/SafeYear3.npy')
X = normalizedata(X)
X1 = np.load('../Data/bankruptYear3.npy')
X1 = normalizedata(X1)
#Y = numpy.load('../../../Data/DataYear1Labels.npy')
Y = np.zeros((X.shape[0],1))
Y1 = np.ones((X1.shape[0],1))
print "DataRead"

# Data Split
X_train, X_test0, y_train, y_test0 = data_split(X, Y, test_siz=0.15, random_state=seed)

X_test = np.concatenate((X_test0,X1),axis=0)
y_test = np.concatenate((y_test0,Y1),axis=0)


X_train, y_train = shuffle(X_train, y_train, random_state=seed)
X_test, y_test = shuffle(X_test, y_test, random_state=seed)

#----------------------------------------------------------------------
print X_train.shape
print X_test.shape

history=History()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate,history]

autoencoder.fit(X_train, X_train,nb_epoch=100,batch_size=64,shuffle=False, callbacks=callbacks_list,validation_data=(X_test, X_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_data = encoder.predict(X_test)
decoded_data= decoder.predict(encoded_data)

accu, y_pred = pred_anomally(X_test,decoded_data,y_test)
print y_pred
print "Accuracy", accu
print "Prediction F1 Score: ", f1_score(y_test, y_pred)
print "ROC_AUC: ", roc_auc_score(y_test,y_pred,average='micro')
misclass(y_test,y_pred)
