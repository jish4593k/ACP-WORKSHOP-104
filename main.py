import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import messagebox

# Read data
df = pd.read_table("D:\Data_Minig\seance4_analyse_descriptive(ACP)\entreprises.txt", sep='\t', header=0)

# Remove unwanted columns
df_clean = df.drop(['Ent', 'ET'], axis=1)

# Centrage et réduction
sc = StandardScaler()
df_new = sc.fit_transform(df_clean)

# ACP
acp = PCA(svd_solver='full')
coordonnes = acp.fit_transform(df_new)

# PyTorch Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Train Linear Regression Model
def train_linear_regression_model(data, input_dim):
    x = torch.tensor(data[:, :-1], dtype=torch.float32)
    y = torch.tensor(data[:, -1], dtype=torch.float32)

    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# Tkinter GUI
def run_linear_regression():
    input_dim = len(df_clean.columns)
    model = train_linear_regression_model(df_clean.values, input_dim)
    messagebox.showinfo("Linear Regression Result", f"Trained Linear Regression Model:\n{model}")

# Plotly scatter plot for visualizing the first two principal components
fig = px.scatter(x=coordonnes[:, 0], y=coordonnes[:, 1], title='Scatter plot of the first two principal components')
fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')
fig.show()

# Tkinter GUI setup
root = tk.Tk()
root.title("PCA and Linear Regression")

button_linear_regression = tk.Button(root, text="Run Linear Regression", command=run_linear_regression)
button_linear_regression.pack()

root.mainloop()
