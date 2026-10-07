print("MAIN SCRIPT STARTED")

from preprocess_data import preprocess_data
from multinomial_naive_bayes import multinomial_naive_bayes

print("Data preprocessing started")
# Preprocess the data
X_train_bow, X_test_bow, y_train, y_test = preprocess_data()

print("MNNB started")
# Run Multinomial Naive Bayes
multinomial_naive_bayes(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)
#new