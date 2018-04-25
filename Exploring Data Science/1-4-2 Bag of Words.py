# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:07:39 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Bag of Words Section 1.4.2

##Objective is to learn how to interpret text into a form that our models can 
#understand.

#Think of each word in a sentence as a symbol.
#Represent it as:
#{'think','of','each','word','in','a','sentence','as','symbol'}
#Notice that we only counted the word 'a' as a single symbol.

#Once we have the set of unique symbols, we need to count how many times each 
#symbol appears in the text. 

#Our example set of terms looks like this:
#1,1,1,1,1,2,1,1

#Notice, each term appears only once, except for the term 'a,' which appears 
#twice. This is called a term vector.

##General Text Processing

#additional steps that need to be taken to get from a sentence to a set of 
#unique terms. main concern is when two symbols are conceptually the same but 
#are not exact string matches. For example, 'house', 'House', and 'HOUSE' are 
#all the same word and represent the same idea. However, these words would not 
#be equal when checked for string equality due to a variance in capitalization. 
#Example:

#from The Hitchhiker's Guide to the Galaxy

#"The Answer to the Great Question... Of Life, the Universe and Everything... 
#Is... Forty-two,' said Deep Thought, with infinite majesty and calm."

#First, we want to strip the punctuation

#"The Answer to the Great Question Of Life the Universe and Everything Is 
#Forty-two said Deep Thought with infinite majesty and calm"

#Then, remove capitalization, which allows us to directly compare terms 
#regardless of their original capitalization

#"the answer to the great question of life the universe and everything is 
#forty-two said deep thought with infinite majesty and calm"

#empty vecotors are uninteresting-no way to differentiate between vectors like that

#To account for this, we need to select terms that are likely to be in multiple 
#sentences. A method typically used to select terms that are unique is "term 
#frequency-inverse document frequency," or tf-idf. This method selects terms 
#that are relatively unique for a particular document by de-emphasizing terms 
#which appear in many documents. 
#Intuitively, terms that are very frequent across many documents like "a", 
#"the", and "they" are not very discriminative. Whereas, a document collection 
#of sports articles may contain inordinately high frequencies of terms like 
#"catcher", "pitcher", "bat" and "base" compared to the "average" document, 
#and even documents about other sports. This gives these terms a higher 
#discriminative power, which is reflected in a higher tf-idf score.

#Tf-idf has two major components
#term frequency is the number of times a term appears in a particular document. 
#tf = frequency of term in document / total number of words in document
#example: if we had a document with 100 words, and 50 of them were the word "bat," 
#the term frequency of "bat" would be 0.5. (Admittedly, this would be a very 
#odd document.)

#inverse document frequency measures how frequently a term appears relative to 
#the number of documents using the inverse of the proportion of documents that 
#contain the term then transformed via logarithm
#idf  = log(number of documents / number of documents with term in it)
#example: if we have a total of 1000 documents, and the word "bat" occurs in 
#750 of them, our idf would be log(1000 / 750) = 0.124

#To get the tf-idf score for a term in a given document, we simply multiply 
#the tf with the idf
#tf-idf = tf*idf = 0.5 * 0.124 = 0.062

#What if our term doesn't occur in as many documents? How does this impact our 
#score? Let's say "bat" only occurs in 10 of our documents
#idf = log(1000/10) = 2
#tf-idf = 0.5 * 2 = 1
#tf-idf score jumped significantly- he term is much more specific for this 
#document than for other documents. Simply, this word is more likely to appear 
#in a document about baseball than in the "average" document

#Once you have a score for each term in each document, you can choose to only 
#include the terms that have a tf-idf score above a threshold

#TL:DR
#If we include every term from every document, we would end up with a really large vector. Some estimates of the number of words in the English language alone are close to one million. This may not sound like a lot, but it would result in a very sparse vector. This means that the vector would be mostly zeros, as most documents wouldn't contain the majority of English terms. A sparse vector creates a very inefficient representation problem for the data and causes other statistical problems; mainly, the Curse of Dimensionality. See the Field Guide to Data Science for more information. By using tf-idf, low scoring terms that occur frequently in other documents are excluded, and do not complicate your model

#Beyond Bags of Words
#A difficulty of the bag of words model is that term context is not used
#We lose the structure of the sentence
#The model also has a lack of understanding of what a term means in context
#Words with multiple definitions

#One very simple hack to incorporate a little bit of context into your Bag of 
#Words Model is to use "n-grams" as features. N-grams are sequences of words of 
#length n: 2-grams(bigrams), 3-grams(trigrams), that appear in a document.
#the phrase "I really like Data Science" would contain the following 2-grams:
#{'i really', 'really like', 'like data', 'data science'}
#n-grams often contain context in a way that might help differentiate between 
#separate uses or meanings of a word. 'river bank' and 'investment bank' could 
#start to help your models realize which type of "bank" you are discussing
#longer n-grams may yield terms that are so specific to particular documents 
#that they are not useful in predicting the meaning of new documents
#Another challenging problems are similar items like forty-two and 42
#Also term endings- photo, photograph, and photography are related-
#Stemming is used to remove endings of words. This can be useful for further 
#collapsing related terms into a single symbol.

#Sometimes, you don't need a very complex model to get the job done. Other 
#times, the added time and cost of deploying a much more complicated model may 
#not yield enough additional functionality to be worth it.


