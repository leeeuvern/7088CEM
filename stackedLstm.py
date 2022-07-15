from pyexpat import model
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing 
from collections import deque
import random
import numpy as np 
import time
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


LengthOfSequence = 60
PredictionFutureLength = 3
PairToPredict = "ethgbp"
EPOCHS = 20
BATCH_SIZE = 64
NAME = f"{LengthOfSequence}-SEQ-{PredictionFutureLength}-PRED-{int(time.time())}"



def classify(current, future_close):
    if float(future_close)>float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future_close', 1)

    for col in df.columns:
        if col!= "action":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
          
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)


    print(df)


    sequential_data = []
    prev_days = deque(maxlen=LengthOfSequence)
  

    for i in df.values:
      
        prev_days.append([n for n in i[:-1]])
        if(len(prev_days)) == LengthOfSequence:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []
 
    for seq, action in sequential_data:
        if action == 0:
            sells.append([seq,action])
        elif action == 1:
            buys.append([seq,action])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
   
    sequential_data = buys+sells
    random.shuffle(sequential_data)

    x = []
    y= []
    
    for seq, action in sequential_data:
        x.append(seq)
        y.append(action)

    return np.array(x), y



ratios = ["btcgbp","ethgbp"]
maindf = pd.DataFrame()
for ratio in ratios:
    dataset = f'{ratio}.csv'

    df = pd.read_csv(dataset, names=["time","open","close","high","low","volume"])
    df.rename(columns={"close": f"{ratio}_close","volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time",inplace=True)
    df = df[[f"{ratio}_close",f"{ratio}_volume"]]
   # print(df.head())


    if len(maindf) == 0:
        maindf = df
    else:
        maindf = maindf.join(df)

maindf.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
maindf.dropna(inplace=True)
maindf['future_close'] = maindf[f"{PairToPredict}_close"].shift(-PredictionFutureLength)

maindf['action'] = list(map(classify,  maindf[f"{PairToPredict}_close"], maindf["future_close"]))

maindf.fillna(method="ffill", inplace=True) 
maindf.dropna(inplace=True)
print(maindf)
print(maindf[[f"{PairToPredict}_close","future_close", "action"]].head(10) )



times = sorted(maindf.index.values)
validationData = times[-int((0.05)*len(times))]
#print(validationData)

validation_main_df = maindf[(maindf.index>= validationData)]
maindf=maindf[maindf.index<validationData]
print(maindf[[f"{PairToPredict}_close","future_close", "action"]].head(10) )


#preprocess_df(maindf)

train_x, train_y = preprocess_df(maindf)
validation_x,validation_y = preprocess_df(validation_main_df)

print(f"train data:  {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys:  {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

#model = Sequential()

input_dim = 28
units = 128
output_size = 10




#model = tf.keras.models.load_model("models/60-SEQ-3-PRED-1657753198")

# Show the model architecture
#model.summary()


model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())

model.add(LSTM(128))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=["accuracy"]
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)


# history = model.fit(
#     train_x, train_y, validation_data=(validation_x, validation_y), batch_size=BATCH_SIZE, epochs=EPOCHS
# )

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))



# history = model.fit(
#     train_x, train_y, validation_data=(validation_x, validation_y), batch_size=BATCH_SIZE, epochs=EPOCHS
# )

#Score model
