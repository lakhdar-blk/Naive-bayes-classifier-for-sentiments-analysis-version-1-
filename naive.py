from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import preprocess_message
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV

sentiments = pd.read_csv("Twitter_Data.csv")

features = ['selected_text']

"""sentiments['sentiment'] = sentiments['sentiment'].map({
    'neutral' : 1,
    'negative' : 0,
    'positive' : 2
    })"""

#sentiments['text'] = sentiments['text'].apply(preprocess_message)

sentiments['selected_text'] = sentiments['selected_text'].apply(preprocess_message)

X = sentiments[features]
Y = sentiments.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


vectorizer = CountVectorizer()
X_train_vc = vectorizer.fit_transform(X_train.selected_text)
X_test_vc = vectorizer.transform(X_test.selected_text)


#transformer = make_column_transformer((vectorizer, 'text'), (vectorizer, 'selected_text'))
#X_train_vc = transformer.fit_transform(X_train)
#X_test_vc = transformer.transform(X_test)

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
              'fit_prior': [True, False],
              'class_prior': [None, [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]}



nb = MultinomialNB()

grid_search = GridSearchCV(nb, param_grid=param_grid, cv=5, verbose=2)

grid_search.fit(X_train_vc, y_train)

#y_pred = nb.predict(X_test_vc)
y_pred = grid_search.predict(X_test_vc)

accuracy = accuracy_score(y_test, y_pred)

# Print accuracy score
print('Accuracy:', accuracy)
print("Confusion matrix:", confusion_matrix(y_test, y_pred))



new_data = np.array([["I can't believe how rude and unprofessional the customer service representative was on the phone just now. They were completely unhelpful and didn't even try to solve my problem. This is not how a company should treat its customers."],
                     ["The new restaurant in town has been getting a lot of buzz lately. I decided to try it out for lunch today and was pleasantly surprised by the quality of the food and the service. The prices were a bit higher than I expected, but overall it was a good experience."],
                     ["I am so grateful for the support and encouragement of my friends and family. They always believe in me and inspire me to pursue my dreams. Without them, I wouldn't be where I am today. Thank you for being there for me!"]])
df = pd.DataFrame(new_data, columns = ['text'])
df['text'] = df['text'].apply(preprocess_message)

print(df['text'])
new_data = vectorizer.transform(df.text)

y_pred = grid_search.predict(new_data)

print("Predicted category:", y_pred)
prediction_interval = grid_search.predict_proba(new_data)

print("Confidence interval",prediction_interval)