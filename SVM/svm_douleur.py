from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pickle
import numpy as np

# Load your annotated data, where sentences are in a list called "sentences" and their labels are in a list called "labels"


# Load the pickle file
with open('annotations_merge_no5.pkl', 'rb') as f:
    annotations = pickle.load(f)

sentences = []
labels = []
for ann in annotations:
    sentences.append(ann[0])
    labels.append(ann[1])

print(labels)
label_scores = {}

def run_svm_train_test(sentences,labels,random_state):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=random_state)

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train the SVM model using GridSearchCV for hyperparameter tuning
    svm = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Train the SVM model using the best hyperparameters
    best_svm = grid_search.best_estimator_
    best_svm.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = best_svm.predict(X_test)

    # Print the classification report and confusion matrix of the model's performance on the testing set
    print("Classification report:")
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print(report)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    label_scores.setdefault('wgt',{})
    label_scores['wgt'].setdefault('wgt_avg_recall',[])
    label_scores['wgt']["wgt_avg_recall"].append(report['weighted avg']['recall'])
    label_scores['wgt'].setdefault('wgt_avg_precison',[])
    label_scores['wgt']["wgt_avg_precison"].append(report['weighted avg']['precision'])
    label_scores['wgt'].setdefault('wgt_avg_f1_score',[])
    label_scores['wgt']["wgt_avg_f1_score"].append(report['weighted avg']['f1-score'])

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
    for i, label in enumerate(best_svm.classes_):
        print(f"Label: {label}")
        label_scores.setdefault(label,{})
        print(f"Precision: {precision[i]}")
        label_scores[label].setdefault('precision',[])
        label_scores[label]['precision'].append(precision[i])
        print(f"Recall: {recall[i]}")
        label_scores[label].setdefault('recall',[])
        label_scores[label]['recall'].append(recall[i])
        print(f"F1-score: {f1_score[i]}")
        label_scores[label].setdefault('f1_score',[])
        label_scores[label]['f1_score'].append(f1_score[i])
        print(f"Support: {support[i]}\n")



for random_state in range(40,45):
    run_svm_train_test(sentences,labels,random_state)

print(label_scores)
print("\nFINAL RESULTS")

for cat in label_scores:
    print('\n'+str(cat))
    for score in label_scores[cat]:
        print(score + ' : '+ str(np.mean(label_scores[cat][score])))




