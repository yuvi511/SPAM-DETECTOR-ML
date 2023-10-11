import os
import io
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, render_template

app = Flask(__name__)

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})
data = pd.concat([data, dataFrameFromDirectory("emails/spam", "spam")])
data = pd.concat([data, dataFrameFromDirectory("emails/ham", "ham")])

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = [request.form['message']]
        example_counts = vectorizer.transform(message)
        prediction = classifier.predict(example_counts)
        result = "Spam" if prediction[0] == 'spam' else "Not Spam"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,port=7000)
