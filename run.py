import sys
import numpy
import pandas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

'''
Spliting the url into bag of words
'''
def extract(data):
    data = list(data)
    for i in range(len(data)):
        if (ord(data[i]) >= ord('a') and ord(data[i]) <= ord('z')) \
            or (ord(data[i]) >= ord('A') and ord(data[i]) <= ord('Z'))\
                or (ord(data[i]) >= ord('0') and ord(data[i]) <= ord('9')):
            continue
        else:
            data[i] = '-'
    return "".join(data).split('-')


non_mal_data = pandas.read_csv('non_malicious_urls.txt', delimiter=',', )
non_mal_data = pandas.DataFrame(non_mal_data)
mal_data = pandas.read_csv('malicious_urls.txt', delimiter=',', )
mal_data = pandas.DataFrame(mal_data)

if len(sys.argv) == 1:
    sys.argv.append('all')

# Handling user custom data length

if sys.argv[1] == 'equal':
    print("Using 1:1, equal amount of data")
    all_data = numpy.concatenate((non_mal_data[:len(mal_data)], mal_data), axis=0)
elif sys.argv[1] == 'double':
    print("using 1:2, double amount of non-malicious data")
    all_data = numpy.concatenate((non_mal_data[:len(mal_data)*2], mal_data), axis=0)
elif sys.argv[1] == 'all':
    print("Using all the data")
    all_data = numpy.concatenate((non_mal_data, mal_data), axis=0)
else:
    print("Using all the data")
    all_data = numpy.concatenate((non_mal_data, mal_data), axis=0)
    print("Invalid argument so choosing default all data!")
all_data = numpy.array(all_data)

test_data = []
url = []

# Handling corrupt data and removing numpy.nan
for temp in all_data:
    if type(temp[0]) != type("string")\
            or type(temp[1]) != type("string"):
        continue
    url.append(temp[0])
    test_data.append(temp[1])


# Vectorizing the bag of words in URL
# vectorization can be done in TfidfVectorizer or CountVectorizer
# Ex:
# vectorizer = CountVectorizer(tokenizer=extract, min_df=1)

vectorizer = TfidfVectorizer(tokenizer=extract, min_df=1)

transform_data = vectorizer.fit_transform(url)

# As we have binary target variable so we use Label Binarizer to achieve that
LB = preprocessing.LabelBinarizer()
LB.fit(test_data)
test_data = LB.transform(test_data).ravel()

training_data, testing_data, target_train_data, target_test_data = train_test_split(transform_data, test_data, test_size=0.3, random_state=42)

# Logistic Regression
lgs = LogisticRegression()
lgs.fit(training_data, target_train_data)
print("Logistic Regression: ", end="")
print(lgs.score(testing_data, target_test_data) * 100)

# Random Forest Classifier
rfr = RandomForestClassifier()
rfr.fit(training_data, target_train_data)
print("Random Forest Classfier: ", end="")
print(rfr.score(testing_data, target_test_data) * 100)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(training_data, target_train_data)
print("Decision Tree Classifier: ", end = "")
print(dtc.score(testing_data, target_test_data) * 100)