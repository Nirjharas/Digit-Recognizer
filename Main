import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import lib.model as md
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical

import time
start_time = time.time()

# Load the data
train = pd.read_csv("./input/train.csv")
print(train.shape)

#Prepare dataset
y = train["label"]
X = train.drop("label", axis = 1)
print(y.value_counts().to_dict())
y = to_categorical(y, num_classes = 10)
del train

X = X / 255.0
X = X.values.reshape(-1,28,28,1)

# Shuffle Split Train and Test from original dataset
seed=2
train_index, valid_index = ShuffleSplit(n_splits=1,
                                        train_size=0.9,
                                        test_size=None,
                                        random_state=seed).split(X).__next__()
x_train = X[train_index]
Y_train = y[train_index]
x_test = X[valid_index]
Y_test = y[valid_index]

# Parameters
epochs = 30
batch_size = 64
validation_steps = 10000

# initialize Model, Annealer and Datagen
model, annealer, datagen = md.init_model()

# Start training
train_generator = datagen.flow(x_train, Y_train, batch_size=batch_size)
test_generator = datagen.flow(x_test, Y_test, batch_size=batch_size)

history = model.fit_generator(train_generator,
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=validation_steps//batch_size,
                    callbacks=[annealer])

score = model.evaluate(x_test, Y_test)
print('Test accuracy: ', score[1])

# Saving Model for future API
model.save('Digits-1.3.0.h5')
print("Saved model to disk")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

## Model predict
test = pd.read_csv("./input/test.csv")
print(test.shape)
test = test / 255
test = test.values.reshape(-1, 28, 28, 1)
p = np.argmax(model.predict(test), axis=1)

print('Base model scores:')
valid_loss, valid_acc = model.evaluate(x_test, Y_test, verbose=0)
valid_p = np.argmax(model.predict(x_test), axis=1)
target = np.argmax(Y_test, axis=1)
cm = confusion_matrix(target, valid_p)
print(cm)

## Preparing file for submission
submission = pd.DataFrame(pd.Series(range(1, p.shape[0] + 1), name='ImageId'))
submission['Label'] = p
filename="keras-cnn-{0}.csv".format(str(int(score[1]*10000)))
submission.to_csv(filename, index=False)

elapsed_time = time.time() - start_time
print("Elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
