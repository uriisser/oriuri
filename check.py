import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Define a sentence to be lemmatized
sentence = "The dogs are barking outside."

# Tokenize the sentence
tokenized_sentence = nltk.word_tokenize(sentence)

# Lemmatize the tokens
lemmatized_sentence = [lemmatizer.lemmatize(token) for token in tokenized_sentence]

# Print the lemmatized sentence
print(lemmatized_sentence)


# Stemming:

stemmer = PorterStemmer()

# Stem the tokens
stemed_sentence = [stemmer.stem(token) for token in tokenized_sentence]

print(stemed_sentence)