import pandas as pd
import numpy as np
import nltk
import re, string, random
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('twitter_samples')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
from nltk.corpus import twitter_samples, stopwords
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.tokenize import word_tokenize



def main():

    # Print all of the tweets within a dataset as strings with the strings() method
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')

    # Begin tokenizing (splitting strings into smaller parts called tokens)
    # tokenize the twitter samples and call the first index as an example
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    print(tweet_tokens)
    print()

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    # Store the dictionaries from the get_tweets_for_model function
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    # Prepare the data for training the NaiveBayesClassifier class
    # Label each tweet as either "positivie" or "negative" then create a dataset by joining them
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    # Use the .shuffle() method to randomaly arrange the data to avoid any bias
    random.shuffle(dataset)

    # Split the code between training and testing the data. 6500 towards training and 3500 towards testing
    train_data = dataset[:6500]
    test_data = dataset[6500:]

    # Train the model using the .train() method
    classifier = NaiveBayesClassifier.train(train_data)

    # Use the .accuracy() method to test the model on the testing data
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    
    # Limit the output to 10
    print(classifier.show_most_informative_features(10))

    # A test to see how the model performs on a custom tweet
    custom_tweet = "I love it when someone's laugh is funnier than the actual joke"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))


# Optinal function to normalize/clean the data in other ways
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# print(lemmatize_sentence(tweet_tokens[0]))


# Remove noise or texts that does not play any vital role
def remove_noise(tweet_tokens, stop_words = ()):
    
    # Create a new list to append the cleaned list to
    cleaned_tokens = []

    # A loop for cleaning the data of hyperlinks, punctuation, and other special characters and symbols
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Optional function to find the most common terms in the whole dataset
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# all_pos_words = get_all_words(positive_cleaned_tokens_list)
# freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))


# Convert tweets from cleaned tokens list to dictionaries with keys as the tokens
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":
    main()