import re
import string 

def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    return text




def remove_stopwords(text):
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)





def resume_stemming(text):
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize


    ps = PorterStemmer()

    protected_words = {
        "python","java","javascript","c++","c#","html","html5","css","sql",
        "mysql","mongodb","react","nodejs","django","flask","tensorflow",
        "pytorch","keras","nlp","ai","ml","devops","aws","azure","gcp",
        "linux","docker","kubernetes"
    }
    if text is None:
        text = ""
    text = str(text)

    tokens = word_tokenize(text.lower())

    stemmed = []
    for w in tokens:
        if w in protected_words:
            stemmed.append(w)
        else:
            stemmed.append(ps.stem(w))

    return " ".join(stemmed)
