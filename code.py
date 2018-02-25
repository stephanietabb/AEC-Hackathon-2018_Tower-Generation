from __future__ import print_function
from numpy import linalg
from sklearn.cross_validation import train_test_split 
from collections import Counter
import math
import random as rn
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.sparse import hstack, coo_matrix, csr_matrix, csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler
import scipy.sparse as sc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import scipy.spatial.distance as ssd
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion

def round_1fp(x):
    return round(x, 1)

def index(x):
    global ng
    global rgs
    return int(round_1fp(x)*10)
    # min_ = min(rgs["xrange"][0]*10, rgs["yrange"][0]*10, rgs["zrange"][0]*10)
    # max_ = max(rgs["xrange"][1]*10, rgs["yrange"][1]*10, rgs["zrange"][1]*10)
    # return max(min(int(round_1fp(x)*10), max_), min_)

def get_location(current_point, next_point):
    global ng
    global rgs
    xmin = rgs["xrange"][0]*10
    ymin = rgs["yrange"][0]*10
    zmin = rgs["zrange"][0]*10
    rl = next_point - current_point
    rl = [index(rl[0])-xmin, index(rl[1])-ymin, index(rl[2])-zmin]
    return int(rl[2]*ng*ng + rl[1]*ng + rl[0])

def get_point(current_point, relative_location):
    global ng
    global rgs
    xmin = rgs["xrange"][0]*10
    ymin = rgs["yrange"][0]*10
    zmin = rgs["zrange"][0]*10
    rl = [0, 0, 0]
    rl[0] = int(relative_location)%ng
    rl[1] = int(int(relative_location)/ng)%ng
    rl[2] = int(int(int(relative_location)/ng)/ng)%ng
    rl[0]+=xmin
    rl[1]+=ymin
    rl[2]+=zmin
    return [current_point[0]+rl[0]*0.1,
            current_point[1]+rl[1]*0.1, 
            current_point[2]+rl[2]*0.1]

def get_data():
    import csv
    shapes = {}
    shape_id = -1
    import glob, os
    for file in glob.glob("points/*.csv"):
        previous_z = 100 
        print(file)
        # if not 'Degrees' in file:
        #     continue
        print(file)
        with open(file, 'rU') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                print(row)
                x = round(float(row[0]), 1)
                y = round(float(row[1]), 1)
                z = round(float(row[2]), 1)
                if abs(z-previous_z) > 3.0:
                    shape_id += 1
                    shapes[shape_id] = []
                shapes[shape_id].append([x, y, z])
                previous_z = z
    for si in shapes:
        shapes[si] = np.array(shapes[si])
        min_ = np.min(shapes[si], axis=0) #centralizing x and y
        mean_ = np.mean(shapes[si], axis=0)
        mean_[2] = min_[2] #starting everything from z=0
        shapes[si] = shapes[si]- mean_
        if np.min(shapes[si], axis=0)[2] !=0:
            print("what happended z should never be zero ", np.min(shapes[si], axis=0))
    print ("shapes are :", shapes)
    return shapes

def get_ranges(shapes):
    
    def myplot(xs, ys):
        plt.plot(xs, ys, 'r-')
        # plt.xlim(-5,5)
        # plt.ylim(-5,5)
        # #plt.draw()
        plt.show()
        import time
        time.sleep(1)

    distances = []
    xdiff = []
    ydiff = []
    zdiff = []
    diffs = []
    cs = ['r--', 'bs', 'g^']
    for shape_id in shapes: 
        data = shapes[shape_id]
        for i in range(3, data.shape[0]-1):
            diff = data[i,:]-data[i+1,:]
            xdiff.append(round(data[i+1,0]-data[i,0], 1))
            ydiff.append(round(data[i+1,1]-data[i,1], 1))
            zdiff.append(round(data[i+1,2]-data[i,2], 1))
            distances.append(round(linalg.norm(data[i,:]-data[i+1,:]), 1))
    xdiff = sorted(list(set(xdiff)))
    ydiff = sorted(list(set(ydiff)))
    zdiff = sorted(list(set(zdiff)))
    xmin = min(xdiff)
    xmax = max(xdiff)
    ymin = min(ydiff)
    ymax = max(ydiff)
    zmin = min(zdiff)
    zmax = max(zdiff)

    return {"xrange":(xmin, xmax), "yrange":(ymin, ymax), "zrange":(zmin, zmax)}

shapes = get_data()
rgs = get_ranges(shapes)
ng = 1000

sanity_check = True
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 10
step = 3
segments = []
nps = [] #next points 
nls = [] #next locations

for shape_id in shapes: 
    data = shapes[shape_id]
    for i in range(0, data.shape[0]-maxlen, step):
        #if linalg.norm(data[i,:]-data[i+1,:]) <= 1:
        segments.append(shapes[shape_id][i: i + maxlen, :])
        nps.append(shapes[shape_id][i + maxlen, :])
        nls.append(get_location(shapes[shape_id][i + maxlen-1, :], shapes[shape_id][i + maxlen, :]))
        if sanity_check:
            lc_ = get_location(shapes[shape_id][i + maxlen-1, :], shapes[shape_id][i + maxlen, :])
            pn_ = get_point(shapes[shape_id][i + maxlen-1, :], lc_)
            if linalg.norm((np.array(pn_) - shapes[shape_id][i + maxlen, :])) >= 0.1/2.0:
                print(pn_, shapes[shape_id][i + maxlen, :])
                #print(np.array(pn_) - shapes[shape_id][i + maxlen, :])

locations = list(set(nls))
print('number of sequences:', len(segments))
print('number of possible locations: ', len(locations))

index_to_loc = dict((i, loc) for i, loc in enumerate(locations))
loc_to_index = dict((loc, i) for i, loc in enumerate(locations))

print('Vectorization...')
X = np.zeros((len(segments), maxlen, 3), dtype=np.float)
y = np.zeros((len(segments), len(locations)), dtype=np.bool)
for i, segment in enumerate(segments):
    for t, point in enumerate(segment[:, :]):
        X[i, t, 0] = point[0]
        X[i, t, 1] = point[1]
        X[i, t, 2] = point[2]
    y[i, loc_to_index[nls[i]]] = 1

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, 3)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(locations)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=9)
    start_index = 0 #np.random.randint(0, len(segments) - maxlen - 1)
    shape_id = np.random.randint(0, len(shapes.keys()))
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        print(shape_id, start_index)
        segment = shapes[shape_id][start_index: start_index + maxlen, :]
        generated = [list(point) for point in segment[:, :]]
        print(generated)
        for p in generated:
            print(p)
        print('*' * 50)
        for i in range(100):
            x = np.zeros((1, maxlen, 3))
            for t, point in enumerate(segment):
                x[0, t, 0] = point[0]
                x[0, t, 1] = point[1]
                x[0, t, 2] = point[2]

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_point = get_point(x[0, maxlen-1, :], index_to_loc[next_index])
            
            generated += [next_point]
            segment = np.vstack((segment[1:, :], np.array([next_point])))
            print(next_point)
            # sys.stdout.write(next_char)
            # sys.stdout.flush()
        print()