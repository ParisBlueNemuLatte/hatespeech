import pandas as pd
import numpy as np
import re
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Step 1: Loading Data
train_df = pd.read_csv('C:\Users\nemuk\Downloads\hateval2019\hateval2019_en_train.csv')
test_df = pd.read_csv('C:\Users\nemuk\Downloads\hateval2019\hateval2019_en_test.csv')
dev_df = pd.read_csv('C:\Users\nemuk\Downloads\hateval2019\hateval2019_en_dev.csv')
# Step 2: Data Preprocessing
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.strip().lower()

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
dev_df['clean_text'] = dev_df['text'].apply(clean_text)

from sklearn import feature_extraction
vector = feature_extraction.text.CountVectorizer(stop_words='english')
new_data = vector.fit_transform(train_df['clean_text'])
print(new_data.toarray().shape)
from sklearn.feature_extraction.text import TfidfVectorizer
# Evalue TF-IDF
tfidf_vectorizer =TfidfVectorizer()
X_train= tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_test= tfidf_vectorizer.transform(test_df['clean_text'])
X_dev = tfidf_vectorizer.transform(dev_df['clean_text'])

X_train = X_train.toarray()
X_test=X_test.toarray()
X_dev=X_dev.toarray()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 1,18248)
X_test = X_test.reshape(X_test.shape[0], 1,18248)
X_dev = X_dev.reshape(X_dev.shape[0], 1,18248)
print(X_train)
y_train1 = train_df['HS']
y_test1 = test_df['HS']
y_dev1 = dev_df['HS']
#from keras.utils import np_utils
from tensorflow.python.keras.utils import np_utils
y_train = np_utils.to_categorical(y_train1,2)
y_test = np_utils.to_categorical(y_test1,2)
y_dev  = np_utils.to_categorical(y_dev1,2)


from keras.layers import Conv1D, Dense, Flatten, Input,add,Bidirectional,SimpleRNN
from keras.models import Model

def build_modelrnn():
    a = Input(shape=(1,18248))
    x=SimpleRNN(32,return_sequences=True)(a)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(inputs=a, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30,validation_data=(X_dev, y_dev), batch_size=32, verbose=2)
    return model

model = build_modelrnn()

pred= model.predict(X_test)
pred1= np.argmax(pred, axis=1)

# Evaluate the model
from sklearn.metrics import accuracy_score
print('Accuracy',accuracy_score(y_test1,pred1))
from sklearn.metrics import recall_score
print('Recall',recall_score(y_test1,pred1, average=None))
from sklearn.metrics import f1_score
print('F1 score',f1_score(y_test1,pred1, average=None))
from sklearn.metrics import precision_score
print('Precision',precision_score(y_test1,pred1, average=None))
print(classification_report(y_test1, pred1))  #评估报告



from sklearn.preprocessing import label_binarize
y_test = label_binarize(y_test, classes=[0,1])
fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()



import pandas as pd
import numpy as np
import re
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Step 1: Loading Data
train_df = pd.read_csv('D:\\桌面\\hateval2019_en_train.csv')
test_df = pd.read_csv('D:\\桌面\\hateval2019_en_test.csv')
dev_df = pd.read_csv('D:\\桌面\\hateval2019_en_dev.csv')
# Step 2: Data Preprocessing
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.strip().lower()

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
dev_df['clean_text'] = dev_df['text'].apply(clean_text)

from sklearn import feature_extraction
vector = feature_extraction.text.CountVectorizer(stop_words='english')
new_data = vector.fit_transform(train_df['clean_text'])
print(new_data.toarray().shape)
from sklearn.feature_extraction.text import TfidfVectorizer
# Evalue TF-IDF
tfidf_vectorizer =TfidfVectorizer()
X_train= tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_test= tfidf_vectorizer.transform(test_df['clean_text'])
X_dev = tfidf_vectorizer.transform(dev_df['clean_text'])

X_train = X_train.toarray()
X_test=X_test.toarray()
X_dev=X_dev.toarray()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 1,18248)
X_test = X_test.reshape(X_test.shape[0], 1,18248)
X_dev = X_dev.reshape(X_dev.shape[0], 1,18248)
print(X_train)
y_train = train_df['HS']
y_test = test_df['HS']
y_dev = dev_df['HS']
from tensorflow.python.keras.utils import np_utils
y_train = np_utils.to_categorical(y_train1,2)
y_test = np_utils.to_categorical(y_test1,2)
y_dev  = np_utils.to_categorical(y_dev1,2)


from keras.layers import Conv1D, Dense, Flatten, Input,add,Bidirectional,SimpleRNN
from keras.models import Model

def build_model():
    a = Input(shape=(1,18248))
    x = Conv1D(64, kernel_size=3,strides=1, padding = 'same',  activation = 'relu')(a)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(inputs=a, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30,validation_data=(X_dev, y_dev), batch_size=32, verbose=2)
    return model

model = build_model()

pred= model.predict(X_test)
pred1= np.argmax(pred, axis=1)

# Evaluate the model
from sklearn.metrics import accuracy_score
print('Accuracy',accuracy_score(y_test1,pred1))
from sklearn.metrics import recall_score
print('Recall',recall_score(y_test1,pred1, average=None))
from sklearn.metrics import f1_score
print('F1 score',f1_score(y_test1,pred1, average=None))
from sklearn.metrics import precision_score
print('Precision',precision_score(y_test1,pred1, average=None))
print(classification_report(y_test1, pred1))  #评估报告



from sklearn.preprocessing import label_binarize
y_test = label_binarize(y_test, classes=[0,1])
fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()



import pandas as pd
import numpy as np
import re
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Step 1: Loading Data
train_df = pd.read_csv('D:\\桌面\\hateval2019_en_train.csv')
test_df = pd.read_csv('D:\\桌面\\hateval2019_en_test.csv')
dev_df = pd.read_csv('D:\\桌面\\hateval2019_en_dev.csv')
# Step 2: Data Preprocessing
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.strip().lower()

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
dev_df['clean_text'] = dev_df['text'].apply(clean_text)

from sklearn import feature_extraction
vector = feature_extraction.text.CountVectorizer(stop_words='english')
new_data = vector.fit_transform(train_df['clean_text'])
print(new_data.toarray().shape)
from sklearn.feature_extraction.text import TfidfVectorizer
# Evalue TF-IDF
tfidf_vectorizer =TfidfVectorizer()
X_train= tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_test= tfidf_vectorizer.transform(test_df['clean_text'])
X_dev = tfidf_vectorizer.transform(dev_df['clean_text'])

X_train = X_train.toarray()
X_test=X_test.toarray()
X_dev=X_dev.toarray()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 1,18248)
X_test = X_test.reshape(X_test.shape[0], 1,18248)
X_dev = X_dev.reshape(X_dev.shape[0], 1,18248)
print(X_train)
y_train = train_df['HS']
y_test = test_df['HS']
y_dev = dev_df['HS']
y_train = np_utils.to_categorical(y_train1,2)
y_test = np_utils.to_categorical(y_test1,2)
y_dev  = np_utils.to_categorical(y_dev1,2)

from keras.layers import Conv1D, Dense, Flatten, Input,add,LSTM
from keras.models import Model

def build_modelrnn():
    a = Input(shape=(1,18248))
    x=LSTM(32,return_sequences=True)(a)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(inputs=a, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30,validation_data=(X_dev, y_dev), batch_size=32, verbose=2)
    return model

model = build_modelrnn()

pred= model.predict(X_test)
pred1= np.argmax(pred, axis=1)

# Evaluate the model
from sklearn.metrics import accuracy_score
print('Accuracy',accuracy_score(y_test1,pred1))
from sklearn.metrics import recall_score
print('Recall',recall_score(y_test1,pred1, average=None))
from sklearn.metrics import f1_score
print('F1 score',f1_score(y_test1,pred1, average=None))
from sklearn.metrics import precision_score
print('Precision',precision_score(y_test1,pred1, average=None))
print(classification_report(y_test1, pred1))  #评估报告



from sklearn.preprocessing import label_binarize
y_test = label_binarize(y_test, classes=[0,1])
fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()