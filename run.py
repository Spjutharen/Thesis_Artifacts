from utilities import *
from detect_adv import *
from cnn import *
import h5py


# Load file.
print("Loading existing file '{}'.".format('data.h5'))
f = h5py.File('data.h5', 'r')
train_data = f['train_data'][:]
train_labels = f['train_labels'][:]
val_data = f['val_data'][:]
val_labels = f['val_labels'][:]
x_test = f['test_data'][:]
y_test = f['test_labels'][:]
if y_test.shape[1] == 10:
    print('{} :OBS: Loaded file not containing Omniglot images :OBS: {}'.format(('='*10), ('='*10)))
else:
    print('{} :OBS: Loaded file contains {} Omniglot images :OBS: {}'.format(('='*10),
                                                                             len(y_test)/2, ('='*10)))
f.close()

le = 400
train_data = train_data[:le]
train_labels = train_labels[:le]
val_data = val_data[:le]
val_labels = val_labels[:le]
x_test = x_test[np.int(len(x_test)/2-le/2):np.int(len(x_test)/2+le/2)]
y_test = y_test[np.int(len(x_test)/2-le/2):np.int(len(x_test)/2+le/2)]


print(y_test.shape)

# Build model.
tf.reset_default_graph()
sess = tf.Session()
model = MnistCNN(sess)
#
# Test model.
preds, _, activations = model.predict(x_test)

# Evaluate with accuracy.
accuracy = np.sum(np.argmax(y_test, 1) == preds)
print('Test accuracy {}'.format(accuracy/len(y_test)))

kdes, lr = create_detector(model, x_train=np.concatenate((train_data,val_data)),
                           y_train=np.concatenate((train_labels,val_labels)),
                           x_test=x_test, y_test=y_test, dataset='mnist')

print('DONE')

