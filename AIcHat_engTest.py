import nltk
import os
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import pyttsx3
import speech_recognition as Sr
import json

# Load the intents data
with open("bali_data.json") as file:
    data = json.load(file)

def UsEr_InPuT():
    yourInput = input("say something: ")
    return yourInput


        
def get_audio():
    r = Sr.Recognizer()

    with Sr.Microphone() as source:
        print("Listening....")
        audio = r.listen(source)
        said = ''
        r.pause_threshold = 0.1
        try:
            print("Recognizing....")
            said = r.recognize_google(audio, language='eng-in')
            print('You said : ', said)
            return said

        except Exception as e:
            print("[+] Exception :" + str(e))
            return UsEr_InPuT()  # Provide a default input or ask the user for input in case of speech recognition failure

def speak1(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice' , voices[0].id)
    engine.setProperty('rate', 145)





    engine.say(text)
    print(text)
    engine.runAndWait()

# Load the preprocessed data
with open("bali_data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Load the trained model
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("bali_model.tflearn")

# Initialize stemmer
stemmer = LancasterStemmer()

# Preprocess input sentence
def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in "?"]
    return tokens

# Convert input sentence into a bag of words
def bag_of_words(sentence, words):
    sentence_words = preprocess(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Get a response from the chatbot
def get_response(user_input):
    results = model.predict([bag_of_words(user_input, words)])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Example usage
def mychats():
    user_input = get_audio()
    response = get_response(user_input)
    speak1(response)
    return mychats()

mychats()
