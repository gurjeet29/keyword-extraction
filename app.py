# importing libraries
from flask import Flask, render_template, request
import pandas as pd
import json
import plotly
import plotly.express as px
import re
import nltk
from rake_nltk import Rake

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        sentence = request.form['message']

    text = sentence.lower()

    # remove tags
    text1 = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text2 = re.sub("(\\d|\\W)+", " ", text1)

    # initialising rake
    r = Rake()

    # Keyword Extraction
    r.extract_keywords_from_text(sentence)

    # getting keyword phrases ranked highest to lowest with scores
    out = r.get_ranked_phrases_with_scores()

    # creating lists for dataframe
    freq = []
    word = []
    for i in range(0, len(out)):
        freq.append(out[i][0])
        word.append(out[i][1])

    # removing repetated word for highlighting
    new_words = []

    for i in range(0, len(word)):
        indi = word[i].split()
        for i in indi:
            if i not in new_words:
                new_words.append(i)
            else:
                continue

    df = pd.DataFrame(list(zip(freq, word)), columns=['FREQ', 'WORD'])

    # Barplot
    fig = px.bar(df, x="WORD", y="FREQ")
    fig.update_layout({
        'paper_bgcolor': 'rgb(210, 231, 255)'
    })

    # Create graphJSON for HTML file
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # converting string to list
    user_input = text2.split()

    return render_template('result.html', graphJSON=graphJSON, new_words=new_words, user_input=user_input, scroll='text')


if __name__ == '__main__':
    app.run(debug=True)
