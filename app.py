from email import message
import os
import glob

import itertools
from collections import defaultdict
from tabnanny import check
import this
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


from gensim.models.tfidfmodel import TfidfModel
from flask import Flask, render_template, request, redirect
from flask import session, url_for

app = Flask(__name__)
app.config["UPLOAD_PATH"] = "uploads"

dir_path = app.config["UPLOAD_PATH"]


def scooter(countfile):
    articles = []
    words_gen = []
    words_tf_idf = []
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
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        # Instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        # Lemmatize all tokens into a new list: lemmatized
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        # list_article
        articles.append(lemmatized)

    # print(articles)
    
    dictionary = Dictionary(articles)

    corpus = [dictionary.doc2bow(a) for a in articles]
   
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
     
    sorted_word_count = sorted(total_word_count.items(),key=lambda w: w[1], reverse=True)
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
        
    doc=[]
    for a in docs:
        doc+=a
        
        
    
    
    tfidf_weights = tfidf[doc]
    sort_tfidf_weights =  sorted(tfidf_weights,key=lambda w: w[1], reverse=True)
    for word_id, word_count in sort_tfidf_weights[:5]:
        words_tf_idf.append([dictionary.get(word_id), word_count])
    
    return (words_gen,words_tf_idf,dictionary,articles)

# print(os.listdir(dir_path), "ssssssssssssssssssss")


@app.route('/', methods=['POST', 'GET'])
def upload():
    
    global words_gen,words_tf_idf,dictionary,articles
    
        
    if request.method == 'POST':
        countfile = []
        for f in request.files.getlist('files'):
            f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
            countfile.append(os.path.join(app.config["UPLOAD_PATH"], f.filename)) 
            
        words_gen,words_tf_idf,dictionary,articles = scooter(countfile)
        
        return render_template('index.html', words_gen=words_gen, words_tf_idf=words_tf_idf)
 
    return render_template('index.html')

@app.route('/find/', methods=['POST'])
def find_func():
    text = request.form.get("find-text")
    message = text,"is not in file"
    
    
    # words_gen,words_tf_idf,dictionary = scooter(tmpfile)
    if dictionary.token2id.get(text):
        message = text,"is in file"  
    
    return render_template('find.html', words_gen=words_gen, words_tf_idf=words_tf_idf,findis = message,articles = articles)
    





if __name__ == '__main__':
    app.run(debug=True)
