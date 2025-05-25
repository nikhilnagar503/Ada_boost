
import numpy as np
from sklearn.tree import DecisionTreeClassifier
class EasyAdaBoost:


        
    def __init__(self,n_estimators=10):
        self.n_estimaters = n_estimators
        self.model =    []
        self.alphas = []

    

    def fit(self,X,y):
        n_sample = X.shape[0]
        weights = np.ones(n_sample)/n_sample



        for _ in range(self.n_estimaters):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X,y,sample_weight=weights)


            predictions = stump.predict(X)

            # Calculate weighted error rate: sum of weights where prediction is wrong
            incorrect = (predictions != y)


            error = np.sum(weights * incorrect)
            error = max(error, 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)

            weights = weights * np.exp(-alpha * y * predictions)

            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)



            self.models.append(stump)
            self.alphas.append(alpha)


    def predict(self,X):
        final_prediction = np.zeros(X.shape[0])

        for model,alpha in  zip(self.model,self,alpha):
            pred = model.predict(X)
            final_prediction+=  alpha*pred

        

        return np.sign(final_prediction).astype(int)



