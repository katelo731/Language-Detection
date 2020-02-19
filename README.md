This is the README file for A0212780L's submission (e0505563@u.nus.edu).

== Python Version ==

I'm using Python Version 3.7.2 for this assignment.

== General Notes about this assignment ==

# Overview
	In this program, I use dictionary structures to store all information about n-gram:

		- `ngram_list`: a dictionary with the key is an n-gram, and the value is the frequency of occurrence among all languages.
		- `ngram_dict`: a dictionary with 3 rows that each row represents a language model. In each language model, the first row contains total n-gram counts, then following some columns of the frequency of seen n-gram in the language.
		- `ngram_prob`: a dictionary with a structure similar to `ngram_dict` while replacing the frequency of seen n-gram with the probability of seen n-gram.

	In build_LM function, read training data line by line and separate it into two pieces: language and sentence, initially. Second, preprocess the sentence under some criteria like doing case-folding, removing punctuation, or adding paddings to both ends. Third, count the n-gram generated by each sentence. If there is an n-gram never appears before, record it to ngram_list. Then store it to ngram_dict under the corresponding language with the key is the n-gram and the value is 1+k (after applying add-k smoothing). Otherwise, increase the frequency in ngram_list and ngram_dict under the corresponding language. Last but not least, build ngram_prob by ngram_dict from known counts of each n-gram and the total size of the n-gram in each language. We then finish building a language model and return the ngram_prob dictionary.

	In test_LM function, read testing data line by line and preprocess it as previously. Here a vector `lang_prob` is to record sentence probability among languages. Probabilities calculated by multiply n-grams in a sentence every time receive a new sentence. To achieve higher accuracy, I use log multiplication instead of simply multiplying them together since probability may become too close to zero to be observed in the end. Finally, return the language of the highest probability. After checking for testing data, I found there may be another language inside. Thus, I set a threshold to detect whether the probability is higher than it. If not, return predicted language as `other` label.

# Experiments
	I tried some combinations to improve prediction accuracy.

	| is_case_folding | remove_punctuation | add_padding | normalize_probability | exist_other_language |   accuracy   |
	| --------------- | ------------------ | ----------- | --------------------- | -------------------- | ------------ |
	|                 |                    |             |                       |                      |  13/20 (65%) |
	|        v        |                    |             |                       |                      |  13/20 (65%) |
	|        v        |         v          |             |                       |                      |  13/20 (65%) |
	|        v        |         v          |      v      |                       |                      |  12/20 (60%) |
	|        v        |         v          |             |           v           |                      |  18/20 (90%) |
	|        v        |         v          |      v      |           v           |           v          |  18/20 (90%) |
	|        v        |         v          |             |           v           |           v          |  19/20 (95%) |

	In a basic version of the program, there is no preprocessing or optimization. After adding case-folding and removing punctuations, the accuracy is still as same as the basic one.
	Adding padding to both ends is applied, but performance dropped a little which is out of my imagination. I suggest due to the training data is not large enough, and padding n-grams set are too small. Thus the model couldn't handle a situation like this.
	There is significant progress after normalizing the probabilities, I use log multiplication instead of a raw one, to prevent many probabilities multiplying together may lead to probability product too close to zero.
	In testing data, two cases belong to other languages. I use a threshold to prevent prediction falls into given language labels like this situation. And it improves the performance a little bit.
	To summarize, the best combination of these variables elevates the accuracy from 65% to 95%. In my opinion, if training data is larger and languages in it are distributed uniformly, the language model would be more powerful to predict.

== Files included with this submission ==

	- `build_test_LM.py`: use a language model constructed by training dataset and output the prediction based on testing data. 

== Statement of individual work ==

[x] I, A0212780L, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

== References ==

Python 3.8.1 documentation (https://docs.python.org/3/tutorial/datastructures.html): usage of dictionaries in python data structure.
