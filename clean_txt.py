import string
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet = True)
  
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
redundant_words = set.union(stop_words, punctuations)

def clean_txt(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_data = ''
    cleaned = []
    for w in word_tokens:
        if w.lower() not in redundant_words:
            filtered_data += lemmatizer.lemmatize(w) + ' '
            cleaned.append(lemmatizer.lemmatize(w))
    return filtered_data, cleaned

print(clean_txt('The, quick) Brown fox; jumps oVer- tHe lazY! d[o]g.'))