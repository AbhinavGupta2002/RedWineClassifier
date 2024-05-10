# Red Wine Classifier

1. Built a **non-linear feedforward deep neural network** with Sigmoid and ReLU activation functions and the Mean Squared Error loss
function to predict the quality of red wine, given its physical attributes - rating can be a decimal from 0 to 10.

2. Trained the model for 500 epochs with 5-fold cross validation on a wine quality dataset using the backpropogation algorithm in batched gradient descent using **Python** with **NumPy**, **Matplotlib**, and **SkLearn** libraries.

3. 4 complete layers were built in the neural network. Each layer had 8, 8, 4, and 1 nodes respectively. Each layer had the ReLU(), ReLU(), Sigmoid(), and Identity() activation function respectively.

4. Average of the mean absolute error (MAE): **0.6907**. This means it is highly accurate (only ~0.7 away from the true rating of a 0 to 10 scale)

5. Standard Deviation of the MAE: **0.033**.

6. Below is a plot generated with **Python** and **Matplotlib** where the x-axis is the epoch number and the y-axis is the average training loss across all experiments for the current epoch.


![Screenshot 2024-05-09 at 9 18 44 PM](https://github.com/AbhinavGupta2002/RedWineClassifier/assets/79180704/623b0e33-60f2-49f2-8985-3f08b8000820)


**NOTE:** Dataset was downloaded from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication).
