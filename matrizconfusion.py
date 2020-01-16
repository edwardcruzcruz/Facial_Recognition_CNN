from sklearn.metrics import confusion_matrix
import numpy as np
import csv
import matplotlib.pyplot as plt

labels=["sorin", "andreea", "alex","javier","jose","jonal","maria","orly"]
Exp = []
Pred = []
with open('csvfile.csv') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         Exp.append(row['Expect'])
         Pred.append(row['Predict'])
         
#print(len(Exp))
test = np.asarray(Exp)
train = np.asarray(Pred)
for elem in test:
    print(elem)
cm=confusion_matrix(Exp, Pred,labels)#, labels	
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
