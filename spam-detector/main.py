import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)


print("\n" + "="*45)
print("     EMAIL SPAM DETECTION SYSTEM")
print("="*45)


print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy        : {accuracy*100:.2f}%")


msg = input("\n📩 Enter your message: ")

msg_vec = vectorizer.transform([msg])
result = model.predict(msg_vec)[0]

# Get probability
prob = model.predict_proba(msg_vec)
spam_prob = prob[0][1] * 100
ham_prob = prob[0][0] * 100


print("\n📩 MESSAGE ANALYSIS")
print(f"Message         : {msg}")

if result == 1:
    print(f"Prediction      : SPAM ❌")
    print(f"Confidence      : {spam_prob:.2f}%")
else:
    print(f"Prediction      : NOT SPAM ✅")
    print(f"Confidence      : {ham_prob:.2f}%")

print("\n" + "="*45)