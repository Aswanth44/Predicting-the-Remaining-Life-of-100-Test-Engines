import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

cols = ["unit", "cycle", "set1", "set2", "set3"] + ["s" + str(i) for i in range(1, 22)]

train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None, names=cols)
true_rul = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

print("Train rows:", len(train), " Test rows:", len(test), " Test engines:", len(true_rul))

train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
train["RUL"] = train["RUL"].clip(upper=125)

sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
X_train = train[sensors]
y_train = train["RUL"]

model = LinearRegression()
model.fit(X_train, y_train)

last = test.groupby("unit").last()
pred = model.predict(last[sensors])

print("R2 score     :", round(r2_score(true_rul["RUL"], pred), 3))
print("Average error:", round(mean_absolute_error(true_rul["RUL"], pred), 1), "cycles")
print("Engine 1  -> true:", true_rul["RUL"][0], " predicted:", round(pred[0]))
print("Engine 2  -> true:", true_rul["RUL"][1], " predicted:", round(pred[1]))

plt.scatter(true_rul["RUL"], pred, s=18)
plt.plot([0, 150], [0, 150], color="red", label="Perfect prediction")
plt.xlabel("True remaining life (cycles)")
plt.ylabel("Predicted remaining life (cycles)")
plt.title("100 test engines: true vs predicted RUL")
plt.legend()
plt.show()
