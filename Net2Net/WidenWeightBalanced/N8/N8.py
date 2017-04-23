import numpy,math
import keras,csv,os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_array
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras.layers import Activation
from keras.utils.np_utils import to_categorical
#from sklearn.metrics import accuracy_score
from keras import metrics
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def normalizedata(X):
	for i in range(X.shape[1]):
#		print (np.max(np.abs(X[:,i])))
		X[:,i] /= (numpy.max(numpy.abs(X[:,i])))
#	print "\n\n"
	return X

def predictUD(yp,cutof):
	p = numpy.zeros((yp.shape[0],1))
	pUO = numpy.zeros((yp.shape[0],1))
	for i in range(yp.shape[0]):
		if yp[i,1] >= cutof:
			p[i] = 1
		else:
			p[i] = 0 

	for i in range(yp.shape[0]):
		if yp[i,1] >= 0.5:
			pUO[i] = 1
		else:
			pUO[i] = 0 
	return p,pUO



def Find_Optimal_Cutoff(target, predicted):
	""" Find the optimal probability cutoff point for a classification model related to event rate
	Parameters
	----------
	target : Matrix with dependent or target data, where rows are observations
	
	predicted : Matrix with predicted data, where rows are observations
	
	Returns
	-------     
	list type, with optimal cutoff value
	
	"""
	fpr, tpr, threshold = roc_curve(target, predicted)
	i = numpy.arange(len(tpr)) 
	roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
	roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
	
	return list(roc_t['threshold']) 
	


# Plot data
def generate_results(y_test, y_score):
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.05])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
#    plt.show()
	plt.savefig('temp.png')
	print('AUC: %f' % roc_auc)


def LoadData():
	X_train = numpy.load('../../../Data/Year1_trainData.npy')
	y_train = numpy.load('../../../Data/Year1_trainLabels.npy')
	X_test = numpy.load('../../../Data/Year1_testData.npy')
	y_test = numpy.load('../../../Data/Year1_testLabels.npy')
	return X_train, y_train, X_test, y_test

def to_categoricalUD(y):
	yc = numpy.zeros((y.shape[0],2))
	j = 0
	for i in range(y.shape[0]):
		if y[i] == 0:
			yc[i,0] = 1
			j += 1
		elif y[i] == 1:
			yc[i,1] = 1
			j += 1
		else:
			yc[i,2] = 1
			j += 1

	return yc

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





def f1sc(y_true, y_pred):
	return f1_score(y_true, y_pred, average='weighted')  

def acc(t,l):
	tru = numpy.argmax(t,1)
	pred = numpy.argmax(l,1)
	acc = numpy.mean(tru == pred)
	return acc

# learning rate schedule
def step_decay(epoch):
        initial_lrate = 0.0001
        drop = 0.5
        epochs_drop = 100.0
        lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
        return lrate




#Load Teacher_Model
teacher_model = load_model('N4_teacher.h5')

# extract Teacher_Model Weights
W1b1 = teacher_model.get_layer('h1').get_weights()
W2b2 = teacher_model.get_layer('out').get_weights()

teacher_W1 = W1b1[0]
teacher_b1 = W1b1[1]
teacher_W2 = W2b2[0]
teacher_b2 = W2b2[1]


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X_train, y_train, X_test, y_test = LoadData()
X_train = normalizedata(X_train)
X_test = normalizedata(X_test)



# create model
model = Sequential()
prelu=keras.layers.advanced_activations.PReLU()
HiddenNeurons = 8
TeacherNeurons = 4

model.add(Dense(HiddenNeurons, input_dim=X_train.shape[1], init='uniform',name='h1'))#, activation='relu'))
model.add(prelu)

model.add(Dense(2, init='uniform',activation='softmax',name='out'))
#model.add(Activation('softmax'))

print "\nW1",teacher_W1.shape
print "\nW2",teacher_W2.shape

########################## Widen operation ##########################
ExtraNeurons = HiddenNeurons-TeacherNeurons
# Randon Idx selection for Neuron Replication
idx = numpy.random.randint(teacher_W1.shape[1],size=ExtraNeurons)
# Replicate input weights of new neurons with weights of neurons in corresponding indices
tmpW1 = teacher_W1[:,idx]
# Add the new neurons to the weight matrix
student_W1 = numpy.concatenate((teacher_W1,tmpW1),axis=1)
# Replicate biases of new neurons with those of neurons in corresponding indices
tmpb1 = teacher_b1[idx]
# Add new neurons to bias vector
student_b1 = numpy.concatenate((teacher_b1,tmpb1))
# Take count of number of neurons of the same type being replicated
scaler = numpy.bincount(idx)[idx] + 1 #******** +1 for already existing neuron
# Scale down output weights according of new neurons by the scaler
tmpW2 = teacher_W2[idx,:]/scaler[:,None]
# Add some white noise to the new weights to enable faster training
noisyW2 = tmpW2+numpy.random.normal(0,1e-4,size=tmpW2.shape)
# Add new noisy neurons to output weight matrix
student_W2 = numpy.concatenate((teacher_W2,noisyW2),axis=0)
# Scale down existing neurons that were replicated.
student_W2[idx,:] = tmpW2
# Equate biases of next layer neurons to that of teacher network
student_b2 = teacher_b2 #******************* No change in output weights

# Set the calculated weights and biases
model.get_layer('h1').set_weights([student_W1, student_b1])
model.get_layer('out').set_weights([student_W2, student_b2])


# Compile model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "FullyConnectedNetworkPrelu.png")
plot(model, to_file=model_path, show_shapes=True)
adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['categorical_accuracy'])

# learning schedule callback
history=History()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate,history]

#model Fitting
print "Training..."
model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=150, batch_size=64, callbacks=callbacks_list, verbose=1)

#Model prediction
predicted=model.predict_proba(X_test,batch_size=25)
pred=model.predict_classes(X_test,batch_size=25)

yt = numpy.argmax(y_test,1)
cutof = numpy.array(Find_Optimal_Cutoff(yt, predicted[:,1]))
print cutof
yp,ypUo = predictUD(predicted,cutof)
#print "Prediction F1 Score: ", f1sc(y_test,y_pred)
#yp = numpy.argmax(y_pred,1)

print "ROC_AUC_OPT: ", roc_auc_score(yt,yp)
print "ROC_AUC: ", roc_auc_score(yt, predicted[:,1])
print "ROC_AUC_Pred: ", roc_auc_score(yt, pred)
misclass(yt,yp)
misclass(yt,ypUo)
misclass(yt,pred)
#generate_results(yt, ypUo)

numpy.save("Prediction.npy",predicted)
numpy.save("Xtest.npy",X_test)
model.save('N8_student.h5')

