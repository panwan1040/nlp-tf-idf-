from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from pyexpat import model
from lib2to3.pgen2 import token
from distutils.filelist import findall
from email import message
from importlib.metadata import requires
import os

import re
import itertools
from collections import defaultdict
from sqlite3 import Row
from tabnanny import check
import this
from turtle import onclick
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import json

from wtforms.form import BaseForm, Form
from wtforms.fields import BooleanField, FormField

from flask_wtf import FlaskForm
from wtforms.form import BaseForm
from wtforms import StringField, SubmitField, BooleanField, FormField, TextAreaField

from flask_bootstrap import Bootstrap

from gensim.models.tfidfmodel import TfidfModel
from flask import Flask, render_template, request, redirect, Markup
from flask import session, url_for

import spacy
from spacy import displacy

from IPython.core.display import display, HTML


from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')


app = Flask(__name__)
app.config["UPLOAD_PATH"] = "uploads"
app.config["SECRET_KEY"] = "rrrrrrraaaaaaaaannnnnnnnddddddddoooooommmmmmmmm"
Bootstrap(app)
dir_path = app.config["UPLOAD_PATH"]

filenamessave = []


def scooter(countfile):
    articles = []
    words_gen = []
    words_tf_idf = []
    docsfor = []

    # countfile = []
    # for path in os.listdir(dir_path):
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         countfile.append(path)
    print(countfile)
    for i in countfile:
        # print(i)
        # f = open(f".\\uploads\\{i}", "r", encoding='utf-8')
        f = open(i, "r", encoding='utf-8')
        article = f.read()

        tokens = word_tokenize(article)

        lower_tokens = [t.lower() for t in tokens]

        alpha_only = [t for t in lower_tokens if t.isalpha()]
        # Remove all stop words: no_stops
        no_stops = [
            t for t in alpha_only if t not in stopwords.words('english')]
        # Instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        # Lemmatize all tokens into a new list: lemmatized
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        # list_article
        articles.append(lemmatized)
        docsfor.append(article)

    # print(articles)

    dictionary = Dictionary(articles)

    corpus = [dictionary.doc2bow(a) for a in articles]

    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count

    sorted_word_count = sorted(
        total_word_count.items(), key=lambda w: w[1], reverse=True)
    for word_id, word_count in sorted_word_count[:5]:
        words_gen.append([dictionary.get(word_id), word_count])
        # print(dictionary.get(word_id), word_count)

    # print(words_gen)
    # tf-idf
    from gensim.models.tfidfmodel import TfidfModel
    tfidf = TfidfModel(corpus)

    docs = []
    for i in corpus:
        docs.append(i)

    doc = []
    for a in docs:
        doc += a

    tfidf_weights = tfidf[doc]
    sort_tfidf_weights = sorted(
        tfidf_weights, key=lambda w: w[1], reverse=True)
    for word_id, word_count in sort_tfidf_weights[:5]:
        words_tf_idf.append([dictionary.get(word_id), word_count])
    doccc = doc
    return (words_gen, words_tf_idf, dictionary, articles, docsfor, total_word_count, doccc)

# print(os.listdir(dir_path), "ssssssssssssssssssss")


class findform(FlaskForm):
    textfindbox = StringField("Find ~~")
    submit = SubmitField("Search")


@app.route('/', methods=['POST', 'GET'])
def upload():
    formfind = findform()
    textfind = ''
    global words_gen, words_tf_idf, dictionary, articles, docsfor, total_word_count, doccc

    if (formfind.validate_on_submit()):
        textfindbox = formfind.textfindbox.data
        word_id = dictionary.token2id.get(textfindbox)

        print(total_word_count[word_id],
              "<---------------------------------------------")  # ผลรวมที่เจอทุกไฟล์
        tmpnamesave = []
        num = 0
        for i in docsfor:
            if textfindbox in i:
                tmpnamesave.append(filenamessave[num])
            num += 1
        print(tmpnamesave)

        countinfile = []
        # print(doccc[word_id])
        for i in docsfor:
            if len(re.findall(textfindbox, i)) > 0:
                countinfile.append(
                    len(re.findall(textfindbox.lower(), i.lower())))

        # total_word_count[word_id]
        return render_template('index.html', words_gen=words_gen, words_tf_idf=words_tf_idf, filenamessave=filenamessave, form=formfind, infilename=tmpnamesave, countinfile=countinfile, total_word_count=sum(countinfile))
        # return redirect(url_for('find_func', textfindbox=session["textfindbox"]))

    if request.method == 'POST':
        countfile = []
        for f in request.files.getlist('files'):
            f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
            filenamessave.append(f.filename)
            countfile.append(os.path.join(
                app.config["UPLOAD_PATH"], f.filename))

        words_gen, words_tf_idf, dictionary, articles, docsfor, total_word_count, doccc = scooter(
            countfile)

        return render_template('index.html', words_gen=words_gen, words_tf_idf=words_tf_idf, form=formfind)

    return render_template('index.html', form=formfind)


@app.route('/find/', methods=['POST', 'GET'])
def find_func():
    text = session["textfindbox"]
    # text = request.form.get("find-text")
    message = text, "is not in file"

    # words_gen,words_tf_idf,dictionary = scooter(tmpfile)
    if dictionary.token2id.get(text):
        message = text, "is in file"

    return render_template('find.html', findis=message, articles=docsfor)


#  form spacy


class spacyform(FlaskForm):

    text = TextAreaField("ป้อนชื่อดิ๊")
    submit = SubmitField("let go!!")
    ch1 = BooleanField("CARDINAL", default='checked')
    ch2 = BooleanField("DATE", default='checked')
    ch3 = BooleanField("EVENT", default='checked')
    ch4 = BooleanField("FAC", default='checked')
    ch5 = BooleanField("GPE", default='checked')
    ch6 = BooleanField("LANGUAGE", default='checked')
    ch7 = BooleanField("LAW", default='checked')
    ch8 = BooleanField("LOC", default='checked')
    ch9 = BooleanField("MONEY", default='checked')
    ch10 = BooleanField("NORP", default='checked')
    ch11 = BooleanField("ORDINAL", default='checked')
    ch12 = BooleanField("ORG", default='checked')
    ch13 = BooleanField("PERCENT", default='checked')
    ch14 = BooleanField("PERSON", default='checked')
    ch15 = BooleanField("PRODUCT", default='checked')
    ch16 = BooleanField("QUANTITY", default='checked')
    ch17 = BooleanField("TIME", default='checked')
    ch18 = BooleanField("WORK_OF_ART", default='checked')


@app.route('/spacy', methods=['POST', 'GET'])
def spacy_():
    form = spacyform()
    text = False
    ent = []
    colors = {'CARDINAL': '#3B7573', 'DATE': '#493770', 'EVENT': '#7F4D85', 'FAC': '#B8587B', 'GPE': '#EEBE8F', 'LANGUAGE': '#925BB3',
              'LAW': '#7055AB', 'LOC': '#3C2B61', 'MONEY': '#FFE7BD', 'NORP': '#FF4F7E', 'ORDINAL': '#8CC63E', 'ORG': '#F02C89', 'PERCENT': '#FB943B',
              'PERSON': '#F4CD26', 'PRODUCT': '#07206D', 'QUANTITY': '#F75959', 'TIME': '#F79D39', 'WORK_OF_ART': '#15BED1'}

    if (form.validate_on_submit()):
        # text = request.form.get("textarea")
        nlp = spacy.load('en_core_web_sm')

        text = form.text.data
        doc = nlp(text)

        if(form.ch1.data):
            ent.append(form.ch1.label.text)
        if(form.ch2.data):
            ent.append(form.ch2.label.text)
        if(form.ch3.data):
            ent.append(form.ch3.label.text)
        if(form.ch4.data):
            ent.append(form.ch4.label.text)
        if(form.ch5.data):
            ent.append(form.ch5.label.text)
        if(form.ch6.data):
            ent.append(form.ch6.label.text)
        if(form.ch7.data):
            ent.append(form.ch7.label.text)
        if(form.ch8.data):
            ent.append(form.ch8.label.text)
        if(form.ch9.data):
            ent.append(form.ch9.label.text)
        if(form.ch10.data):
            ent.append(form.ch10.label.text)
        if(form.ch11.data):
            ent.append(form.ch11.label.text)
        if(form.ch12.data):
            ent.append(form.ch12.label.text)
        if(form.ch13.data):
            ent.append(form.ch13.label.text)
        if(form.ch14.data):
            ent.append(form.ch14.label.text)
        if(form.ch15.data):
            ent.append(form.ch15.label.text)
        if(form.ch16.data):
            ent.append(form.ch16.label.text)
        if(form.ch17.data):
            ent.append(form.ch17.label.text)
        if(form.ch18.data):
            ent.append(form.ch18.label.text)

        options = {"ents": ent, "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        return render_template('spacy.html', show_=Markup(html), text=text, form=form)
    return render_template('spacy.html', form=form, text=text)


# fakenews


class fakeform(FlaskForm):
    text = TextAreaField("Text for check")
    submit = SubmitField("Let's go!!")


model_path = "fake-news-bert-base-uncased-minidataset"
max_length = 512
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def get_prediction(text, convert_to_label=False):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())


@app.route('/fakenews', methods=['POST', 'GET'])
def fakecheck():
    formfake = fakeform()
    if (formfake.validate_on_submit()):
        textfindbox = formfake.text.data
        predix = get_prediction(textfindbox)
        if predix == 0:
            predix = "reliable"
        else:
            predix = "fake"

        return render_template('fakenews.html', formfake=formfake, predix=predix)
    return render_template('fakenews.html', formfake=formfake)


class sentform(FlaskForm):
    text = TextAreaField("Text for check", id="todo")
    submit = SubmitField("Let's go!!", id="subsent")


@app.route('/sentiment', methods=['POST', 'GET'])
def senttimentAnalyzer():
    formsent = sentform()
    if (formsent.validate_on_submit()):
        textfindbox = formsent.text.data
        sentizer = SentimentIntensityAnalyzer()
        scores = sentizer.polarity_scores(textfindbox)
        print(scores["neg"], "<---------sent")
        predix = "negative"
        maxv = scores["neg"]
        if(maxv < scores['neu']):
            maxv = scores['neu']
            predix = "neutral"
            if(maxv < scores['pos']):
                maxv = scores['pos']
                predix = "positive"

        return render_template('vader.html', formsent=formsent, predix=predix)
    return render_template('vader.html', formsent=formsent)


# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
