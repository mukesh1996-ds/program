#import string

import pandas as pd
df = pd.read_csv("D:\\data science\\dataset\\spam.csv")
print(df)

# pre processing
# In this step we are using a map function for mapping the value
df[type] = df.type.map({'ham': 0, 'spam': 1})
print(df[type])

# In this function we are again using map function hear we are converting our entire data in x with the help of
# syntax called lambda and then into lower case
df['text'] = df.text.map(lambda x:x.lower())
print(df['text'])
import nltk
nltk.download()
