import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize


# parameters

radius = 3                  # LBP radius
n_neighbours = 8*radius     # LBP number of neighbours
n_classes = 5               # number of classes for k-means

matplotlib.rcParams.update({'font.size': 8})

# functions

def get_set_dict(folder):
    images = {}
    path = os.path.join(os.getcwd(), folder)

    for filename in os.listdir(path):
        if not filename.startswith('.'): # ignore hidden files
            split = filename.split('_')
            images['{}/{}'.format(folder, filename)] = int(split[0])
    
    return images


def generate_histogram(image):
    im = cv2.imread(image)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, n_neighbours, radius, method='uniform')
    x = itemfreq(lbp.ravel())
    hist = x[:, 1]/sum(x[:, 1]) #normalize

    return hist


def generate_plot(ncols, results, title=None, figsize=(8,3.5)):
    nrows = int(len(results) / ncols)
    if len(results) % ncols != 0:
        nrows = nrows + 1 

    fig, axes = plt.subplots(nrows,ncols)
    fig.set_size_inches(figsize)
    # if title:
    #     fig.suptitle(title)
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            if i < len(results):
                axes[row][col].imshow(cv2.cvtColor(cv2.imread(results[i][0]), cv2.COLOR_BGR2RGB))
                axes[row][col].set_title(results[i][1])
            axes[row][col].axis('off')
            i+=1
    filename = ''.join(e for e in title if e.isalnum())
    # plt.show()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


# supervised classification

training_set = get_set_dict('train')
test_set = get_set_dict('test')

X = []
X_files = []
# y = []

for image in training_set:
    hist = generate_histogram(image)
    X.append(hist)
    X_files.append(image)
    # y.append(training_set[image])

for image in test_set:
    hist = generate_histogram(image)
    results = []

    for index, x in enumerate(X):
        score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.HISTCMP_CHISQR)
        results.append((X_files[index], round(score, 3)))
    results = sorted(results, key=lambda el: el[1])
    generate_plot(5, results, 'scores for {}'.format(image))


# unsupervised classification

image_set = dict(training_set, **test_set)
X = []
X_files = []

for image in image_set:
    hist = generate_histogram(image)
    X.append(hist)
    X_files.append(image)

X = np.float32(X)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(X, n_classes, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

results = list(zip(X_files, label.ravel()))
results = sorted(results, key=lambda el: el[1])

generate_plot(3, results, 'K-means classification 1', (6,10.5))

