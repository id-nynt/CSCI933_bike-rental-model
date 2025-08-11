# INFORMATION
print("ASSIGNMENT 1")
print("Student: Ngoc Yen Nhi Tran")
print("CSCI933, SN:8852303, nynt901@uowmail.edu.au")
print()

# --- IMPORT LIBRARY ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# Import addition module from sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import root_mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# Import visualization tools: matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns


# --- EXPERIMENT SETUP ---
# Load dataset
df = pd.read_csv("bike_rental_data.csv")


# Data structure
print("---DATASET---")
df.head()


# Data summary of numerical attributes
df.describe()


# Data description
df.info()


# Plot histogram
df.hist(bins=50, figsize=(20,15))
plt.show()


# Features and target
X = df.drop(columns=["bikes_rented"])
y = df["bikes_rented"]


# Create cyclical features
X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

# Update feature lists
num_features = ["temp", "humidity", "windspeed", "hour_sin", "hour_cos"]
cat_features = ["season"]  # hour is no longer in cat_features
binary_features = ["holiday", "workingday"]

# Redefine ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(), cat_features),
    ("binary", "passthrough", binary_features)
])




# --- DATA SPLIT ---
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# Check the data after preprocessing
# Get feature names from the transformer
feature_names = (
    preprocessor.named_transformers_["num"].get_feature_names_out(num_features).tolist()
    + preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features).tolist()
    + binary_features  # passthrough columns retain original names
)

# Convert to DataFrame
X_train_df = pd.DataFrame(X_train_preprocessed.toarray() if hasattr(X_train_preprocessed, "toarray") else X_train_preprocessed,
                          columns=feature_names)

# Show preview
print("---Data preprocessing---")
X_train_df.head()




# --- VISUALIZATION SETUP ---
# --- Predicted vs. Actual Plot ---
def plot_pred_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r', lw=2)  # 45-degree line
    plt.title(f'Actual vs. Predicted - {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

# --- RMSE & R² Score Bar Chart ---
def plot_model_performance(models, rmse_scores, r2_scores):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE Comparison
    ax[0].bar(models, rmse_scores, color='skyblue')
    ax[0].set_title('RMSE Comparison')
    ax[0].set_ylabel('RMSE (Lower is Better)')
    
    # R² Comparison
    ax[1].bar(models, r2_scores, color='lightcoral')
    ax[1].set_title('R² Comparison')
    ax[1].set_ylabel('R² Score (Higher is Better)')
    
    plt.show()
results = []




# Evaluate models
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred) # Use root_mean_squared_error
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.2f}, R^2={r2:.2f}")

    return {
        "model": name,
        "rmse": rmse,
        "r2": r2
    }



# --- STATISTICAL LEARNING ---
print("--- STATISTICAL LEARNING ---")
# Linear Regression
print("---Linear Regression---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_preprocessed, y_train)
evaluate(lin_reg, X_test_preprocessed, y_test, "Linear Regression")
print()


# Linear Regression Visualization
y_pred_lin = lin_reg.predict(X_test_preprocessed)
plot_pred_vs_actual(y_test, y_pred_lin, "Linear Regression")


# Ridge Regression
print("---Ridge Regression---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_preprocessed, y_train)

# Ridge Regression with Cross-Validation
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]} # Tune lambda
grid_search_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search_ridge.fit(X_train_preprocessed, y_train)

ridge_tuned = grid_search_ridge.best_estimator_

print(f"Best Ridge Alpha: {grid_search_ridge.best_params_['alpha']}")
evaluate(ridge, X_test_preprocessed, y_test, "Ridge Regression (alpha=1.0)")
evaluate(ridge_tuned, X_test_preprocessed, y_test, "Ridge Regression (Tuned)")
print()


# Ridge Regression Visualization
y_pred_ridge = ridge_tuned.predict(X_test_preprocessed)
plot_pred_vs_actual(y_test, y_pred_ridge, "Ridge Regression")


# Lasso Regression
print("---Lasso Regression---")
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_preprocessed, y_train)

# Lasso Regression with Cross-Validation
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_root_mean_squared_error')
grid_search_lasso.fit(X_train_preprocessed, y_train)

lasso_tuned = grid_search_lasso.best_estimator_

print(f"Best Lasso Alpha: {grid_search_lasso.best_params_['alpha']}")
evaluate(lasso, X_test_preprocessed, y_test, "Lasso Regression (alpha=0.1)")
evaluate(lasso_tuned, X_test_preprocessed, y_test, "Lasso Regression (Tuned)")
print()

# Compare Sparsity
ridge_non_zero = (ridge_tuned.coef_ != 0).sum()
lasso_non_zero = (lasso_tuned.coef_ != 0).sum()
print(f"Ridge Non-Zero Coefficients: {ridge_non_zero}")
print(f"Lasso Non-Zero Coefficients: {lasso_non_zero}")
print()


# Lasso Regression Visualization
y_pred_lasso = lasso_tuned.predict(X_test_preprocessed)
plot_pred_vs_actual(y_test, y_pred_lasso, "Lasso Regression")


# Elastic Net
print("---Elastic Net---")
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train_preprocessed, y_train)

# Elastic Net with Cross-Validation
param_grid_elastic = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                       'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
grid_search_elastic = GridSearchCV(elastic_net, param_grid_elastic, cv=5, scoring='neg_root_mean_squared_error')
grid_search_elastic.fit(X_train_preprocessed, y_train)

elastic_net_tuned = grid_search_elastic.best_estimator_

print(f"Best Elastic Net Alpha: {grid_search_elastic.best_params_['alpha']}")
print(f"Best Elastic Net L1 Ratio: {grid_search_elastic.best_params_['l1_ratio']}")
evaluate(elastic_net, X_test_preprocessed, y_test, "Elastic Net (alpha=0.1, l1_ratio=0.5)")
evaluate(elastic_net_tuned, X_test_preprocessed, y_test, "Elastic Net (Tuned)")
print()


y_pred_elastic = elastic_net_tuned.predict(X_test_preprocessed)
plot_pred_vs_actual(y_test, y_pred_elastic, "Elastic Net")


# Statistical Approach Results Summary
print("---Statistical Approach Results Summary---")
result = evaluate(lin_reg, X_test_preprocessed, y_test, "Linear Regression")
results.append(result)

result = evaluate(ridge, X_test_preprocessed, y_test, "Ridge Regression (alpha=1.0)")
result = evaluate(ridge_tuned, X_test_preprocessed, y_test, "Ridge Regression (Tuned)")
results.append(result)

result = evaluate(lasso, X_test_preprocessed, y_test, "Lasso Regression (alpha=0.1)")
result = evaluate(lasso_tuned, X_test_preprocessed, y_test, "Lasso Regression (Tuned)")
results.append(result)

result = evaluate(elastic_net, X_test_preprocessed, y_test, "Elastic Net (alpha=0.1, l1_ratio=0.5)")
result = evaluate(elastic_net_tuned, X_test_preprocessed, y_test, "Elastic Net (Tuned)")
results.append(result)
print()



# --- DEEP LEARNING APPROACH ---
print("--- DEEP LEARNING APPROACH ---")
# Baseline Linear Neural Network model
print("---Baseline Linear Neural Network model---")
class LinearNN(nn.Module):
    def __init__(self, input_dim):
        super(LinearNN, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single-layer linear model
    
    def forward(self, x):
        return self.linear(x)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_preprocessed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_preprocessed, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize model, loss, and optimizer
model = LinearNN(X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # L2 regularization (weight decay)

# Train model 100 epochs
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluate on test data
model.eval()
y_pred_tensor = model(X_test_tensor).detach().numpy()
test_rmse = root_mean_squared_error(y_test, y_pred_tensor) # use root_mean_squared_error
test_r2 = r2_score(y_test, y_pred_tensor)
name = "Deep Learning Model (100 epochs)"
print(f"{name}: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}\n")
results.append({"model": name, "rmse": test_rmse, "r2": test_r2})


# --- Run Visualizations for Deep Learning Model ---
plot_pred_vs_actual(y_test, y_pred_tensor.flatten(), "Deep Learning Model 100 epochs")


# Train model 1000 epochs
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluate on test data
model.eval()
y_pred_tensor = model(X_test_tensor).detach().numpy()
test_rmse = root_mean_squared_error(y_test, y_pred_tensor) # use root_mean_squared_error
test_r2 = r2_score(y_test, y_pred_tensor)
name = "Deep Learning Model (1000 epochs)"
print(f"{name}: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}\n")
results.append({"model": name, "rmse": test_rmse, "r2": test_r2})


# --- Run Visualizations for Deep Learning Model ---
plot_pred_vs_actual(y_test, y_pred_tensor.flatten(), "Deep Learning Model 1000 epochs")




print("---Linear Neural Network with weigh decay---")
# Try different weight decay values
for wd in [0.001, 0.01, 0.1]:
    # Initialize model, loss, and optimizer
    model = LinearNN(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)  # L2 regularization (weight decay)
    
    # Train model
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Evaluate on test data
    model.eval()
    y_pred_tensor = model(X_test_tensor).detach().numpy()
    test_rmse = root_mean_squared_error(y_test, y_pred_tensor) # use root_mean_squared_error
    test_r2 = r2_score(y_test, y_pred_tensor)
    name = "Deep Learning Model (weight decay: " + str(wd) +")"
    print(f"{name}: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}\n")
    results.append({"model": name, "rmse": test_rmse, "r2": test_r2})



# Dropout Regularization 
print("---Linear Neural Network with Dropout---")
# --- Linear Neural Network with Dropout ---
class LinearNNDropout(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(LinearNNDropout, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single-layer linear model
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout Regularization
    
    def forward(self, x):
        x = self.dropout(x)  # Apply dropout
        return self.linear(x)

# Train and evaluate the model for various dropout rate
for dropout_rate in [0.2, 0.5, 0.8]:
    # Initialize model, optimizer, and loss function
    model = LinearNNDropout(X_train_tensor.shape[1], dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # --- Train model ---
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # --- Evaluate on test data ---
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor).numpy()
    
    # --- Compute evaluation metrics ---
    test_rmse = root_mean_squared_error(y_test, y_pred_tensor) 
    test_r2 = r2_score(y_test, y_pred_tensor)
    
    name = "Deep Learning Model with Dropout (" + str(dropout_rate) +")"
    print(f"{name}: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}\n")
    results.append({"model": name, "rmse": test_rmse, "r2": test_r2})



print("---Linear Neural Network with Feature Engineering and Data Augmentation---")
# --- Feature Engineering (Polynomial Features) ---
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_preprocessed)
X_test_poly = poly.transform(X_test_preprocessed)

# --- Data Augmentation: Adding Gaussian Noise to Inputs ---
def add_gaussian_noise(X, mean=0, std=0.1):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

X_train_augmented = add_gaussian_noise(X_train_poly)
X_test_augmented = add_gaussian_noise(X_test_poly)

# Convert target to NumPy array
y_train_array = y_train.values if hasattr(y_train, "values") else y_train
y_test_array = y_test.values if hasattr(y_test, "values") else y_test

# --- Convert to PyTorch tensors ---
X_train_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_augmented, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32).view(-1, 1)

# --- Initialize Model, Loss, and Optimizer ---
model = LinearNN(X_train_augmented.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)  # No weight decay
criterion = nn.MSELoss()

# --- Train Model ---
epochs = 1000
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# --- Evaluate on Test Data ---
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor).numpy()

# --- Compute RMSE & R² Score ---
test_rmse = root_mean_squared_error(y_test, y_pred_tensor)
test_r2 = r2_score(y_test, y_pred_tensor)

name = "Deep Learning Model with Feature Engineering & Data Augmentation"
print(f"{name}: RMSE={test_rmse:.2f}, R^2={test_r2:.2f}\n")
results.append({"model": name, "rmse": test_rmse, "r2": test_r2})



print("RESULT SUMMARY")
print("{0:<65}{1:>10}{2:>10}".format("Model Name", "RMSE", "R^2"))
print("{0:<65}{1:>10}{2:>10}".format("----------", "----", "---"))
for i in results:
    print("{0:<65}{1:>10.2f}{2:>10.2f}".format(i['model'], i['rmse'], i['r2']))


# Compare RMSE & R^2 scores
model_names = ["Linear\nReg.", "LNN\n100e", "LNN\n1000e", "LNN\nwd=0.01", "LNN\ndr=0.5", "LNN\nFeaEng"]

selected_models = ['Linear Regression', 'Deep Learning Model (100 epochs)', 'Deep Learning Model (1000 epochs)', 'Deep Learning Model (weight decay: 0.01)', 'Deep Learning Model with Dropout (0.5)', 'Deep Learning Model with Feature Engineering & Data Augmentation']
filtered_results = [r for r in results if r['model'] in selected_models]

rmse_values = [r['rmse'] for r in filtered_results]
r2_values = [r['r2'] for r in filtered_results]

plot_model_performance(model_names, rmse_values, r2_values)