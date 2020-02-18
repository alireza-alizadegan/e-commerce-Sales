import keras
import numpy as np 
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import mae
import os
import pandas as pd 
import locale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb

def model(dim, regress=True):

    model = Sequential()
    model.add(Dense(512, input_dim=dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    if regress:
        model.add(Dense(1, activation='linear'))
    
    return model

def load_data(inputPath):
    chemours_df = pd.read_csv(inputPath, usecols=['emea', 'la', 'apac', 'fchem_product', 'value_to_USD', 'oil_price'])
    print(chemours_df.columns.values)

    return chemours_df
def preprocess_chemours_data(chemours_df, train, test):

    check_first = True

    # zipBin = LabelBinarizer().fit(chemours_df['region'])
    # regionCate = zipBin.transform(train['region'])
    # testRegCate = zipBin.transform(test['region'])

    # scaler = MinMaxScaler()
    # fchemCon = scaler.fit_transform(train['fchem'])
    # fchemCon = scaler.fit_transform(test['fchem'])
    continous = ['value_to_USD','oil_price']

    scaler = MinMaxScaler()
    continous_ = scaler.fit_transform(train[continous])
    test_con = scaler.transform(test[continous])

    for col in list(train.columns.values):

        if check_first and col != 'fchem_product' and col not in continous:
                
            trainX = train[col]
            testX = test[col]
            check_first = False

        elif not check_first and col != 'fchem_product' and col not in continous:

            trainX = np.vstack([trainX, train[col]])
            testX = np.vstack([testX, test[col]])
    
    trainX = np.vstack([trainX, np.transpose(continous_)])
    testX = np.vstack([testX, np.transpose(test_con)])

    return (trainX, testX)

#Load + split data
print('Loading data...')
df = load_data('fchem.csv')

(train, test) = train_test_split(df, test_size=0.1, shuffle=False)

print('Train:', train.shape)
print('Test:', test.shape)

#Turn training data to numpy
# print('Preprocessing data...')
trainX, testX = preprocess_chemours_data(df, train, test)
trainX = np.transpose(trainX)
testX = np.transpose(testX)

#Normalize values to 0~1
trainY = train['fchem_product'] / train['fchem_product'].max()
testY = test['fchem_product'] / train['fchem_product'].max()

# train_data = train
# train_data['Target'] = train['fchem_product']

# C_mat = train_data.corr()
# fig = plt.figure(1)

# sb.heatmap(C_mat, vmax = .9, square = True)
# plt.show()

print('Train Y', trainY.shape)
print('Test Y', testY.shape)
print(trainY[:1,])
print(trainX[:1,])
print(train, test)
NUM_EPOCHS = 30
BS = 8

#Build model
print('Build model...')
model = model(trainX.shape[1], regress=True)
# optimizer = Adam(lr=1e-4)
optimizer = Adam(lr=1e-3, decay=1e-3/200)
# optimizer = Adam(lr=3e-3)
model.compile(optimizer, loss='mean_absolute_percentage_error')
print(model.summary())
model.save('fchem_model.model')
#Training
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=NUM_EPOCHS, batch_size=BS)

print('Predicting sales ...')
preds = model.predict(testX)
print(preds * train['fchem_product'].max())

diff = preds.flatten() - testY
pDiff = (diff/testY) * 100
abPDiff = np.abs(pDiff)

mean = np.mean(abPDiff)
std = np.std(abPDiff)

print(diff * train['fchem_product'].max())
 
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. sales: {}, std sales: {}".format(
	locale.currency(df["fchem_product"].mean(), grouping=True),
	locale.currency(df["fchem_product"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure(1)
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("Curves_Plot.png")


preds = model.predict(np.concatenate([trainX, testX]))
# preds = model.predict(testX)
# print((np.concatenate([trainX, testX])).shape)

print(testX.shape, trainX.shape)
plt.figure(2)
plt.plot(preds * train['fchem_product'].max())
plt.plot(df['fchem_product'])
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()
plt.savefig('Prediction vs GroundTruth')
# print(preds * train['fchem_product'].max())