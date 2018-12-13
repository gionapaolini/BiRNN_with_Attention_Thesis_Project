import re
import html
from collections import Counter
import numpy as np


def preprocess(text):
    
    text = html.unescape(text)
    
    text = text.lower()
    
    text = text.replace(u'\xa0', u' ')
    text = text.replace('[cr lf]',' ')
    text = text.replace('\u200b','')
    text = text.replace('*','')
        
    text = re.sub("\%\{dialogoption\([A-Za-z0-9 \-\_\'\?\"\(\)\&\/\,\.]*\)\}", " <DIALOG_OPTION> ",text)
    text = re.sub("%{[0-9a-z()]*}", "",text)
   
    text = re.sub("[0-9]{4,5} [0-9]{2,4} [0-9]{2,4}( [0-9]{2,3})?", " <PHONE_NUMBER> ", text)
    text = re.sub("(^[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+$)", " <EMAIL_ADDRESS> ",text)
    
    text = re.sub("\<a [ a-z0-9=`:/.\-\_\?\&\#\(\)\]\[\@]*>", "", text)
    text = re.sub("\<\/a>", "", text)
    
    text = re.sub("[0-9]{1,2}:*([0-9]{2})*(am|pm)", " <HOUR_TIME> ", text)
    
    text = re.sub("www\.[a-z0-9]+\.(com|net|nl|it|eu|org|de)", " <URL> ",text)

    text = re.sub('[0-9]+',' <1_OR_MORE_DIGITS> ', text)
    text = re.sub('( ){2,}',' ', text)
 
    # Replace punctuation with tokens so we can use them in our model
    text = text.replace('i\'m', 'i am')
    text = text.replace('you\'re', 'you are')
    text = text.replace('we\'re', 'we are')
    text = text.replace('they\'re', 'they are')
    text = text.replace('don\'t', 'do not')
    text = text.replace('doesn\'t', 'does not')
    text = text.replace('i\'ll', 'i will')
    text = text.replace('you\'ll', 'you will')
    text = text.replace('it\'ll', 'it will')
    text = text.replace('we\'ll', 'we will')
    text = text.replace('they\'ll', 'they will')
    text = text.replace('who\'ll', 'who will')
    text = text.replace('isn\'t', 'is not')
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('\'', ' <APOSTROPHE> ')
    text = text.replace('’', ' <APOSTROPHE> ')
    text = text.replace('`', ' <APOSTROPHE> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('-', ' <DASH> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('/', ' <FORWARD_SLASH> ')
    text = text.replace('\\', ' <BACKWARD_SLASH> ')
    text = text.replace(']', ' <RIGHT_SQ_PAREN> ')
    text = text.replace('[', ' <LEFT_SQ_PAREN> ')
    text = text.replace(':', ' <COLON> ')
   
    
    return text

	
	
	
def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab,1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
	
	
def get_batches_rnn(n_sentences_in_batch, sentences, articleIds):  

    for idx in range(0, len(sentences), n_sentences_in_batch):
        x_batch = sentences[idx:idx+n_sentences_in_batch]
        y_batch = articleIds[idx:idx+n_sentences_in_batch]
        maxLength = max(len(x) for x in x_batch)
        
        x = np.zeros((len(x_batch),maxLength), dtype=int)
        
        for batchIndex in range(len(x_batch)):
            sentenceLen = len(x_batch[batchIndex])
            for sentenceIndex in range(sentenceLen):
                x[batchIndex,sentenceIndex] = x_batch[batchIndex][sentenceIndex]
				
        yield x, y_batch, maxLength