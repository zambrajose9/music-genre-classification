import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar los datos
def load_data(csv_path='data/raw/features_30_sec.csv'):
    data = pd.read_csv(csv_path)
    X = data.drop(columns=['filename', 'length', 'label'])
    y = data['label']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_one_hot = np.eye(10)[y]  # Convertir las etiquetas a one-hot encoding
    return X, y_one_hot

# Función para dividir los datos
def split_data(X, y, test_size=0.3, val_size=0.5):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para convertir DataFrame o numpy array a tensor
def convert_to_tensor(X, y):
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y)
    return X_tensor, y_tensor

# Obtener la función de activación
def get_activation_function(name):
    activations = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Softmax': nn.Softmax(dim=1)
    }
    return activations.get(name, nn.ReLU())  # Por defecto usa ReLU

# Obtener la función de pérdida
def get_loss_function(name):
    losses = {
        'MSELoss': nn.MSELoss(),
        'CrossEntropyLoss': nn.CrossEntropyLoss()
    }
    return losses.get(name, nn.CrossEntropyLoss())  # Por defecto usa CrossEntropy

# Construir el modelo flexible con capas ocultas y activación
def build_model(config):
    layers = []
    input_size = config['input_size']
    hidden_layers = config.get('hidden_layers', [])
    output_size = config['output_size']
    activation_fn = get_activation_function(config['activation_function'])

    prev_size = input_size
    for hidden_size in hidden_layers:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(activation_fn)
        prev_size = hidden_size

    layers.append(nn.Linear(prev_size, output_size))

    if config['loss_function'] == 'MSELoss':
        layers.append(nn.Sigmoid())  # Para MSE usa Sigmoid
    else:
        layers.append(nn.Softmax(dim=1))  # Para CrossEntropy usa Softmax

    return nn.Sequential(*layers)

# Función de entrenamiento que incluye visualización de pérdidas
def train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config):
    criterion = get_loss_function(config['loss_function'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    epochs = config['epochs']

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        y_train_pred = model(X_train_tensor)
        loss_train = criterion(y_train_pred, y_train_tensor)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validación
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            loss_val = criterion(y_val_pred, y_val_tensor)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        # Imprimir cada 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss Entrenamiento: {loss_train.item():.4f}, Loss Validación: {loss_val.item():.4f}')

    # Graficar las pérdidas de entrenamiento y validación
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Pérdida de Entrenamiento')
    plt.plot(val_losses, label='Pérdida de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Loss)')
    plt.title('Pérdida durante el Entrenamiento y Validación')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Función para evaluar el modelo y calcular métricas
def evaluate_model(model, X_test_tensor, y_test_tensor, config):
    model.eval()
    criterion = get_loss_function(config['loss_function'])

    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        loss_test = criterion(y_test_pred, y_test_tensor)

    print(f'Pérdida en el conjunto de prueba: {loss_test.item():.4f}')

    y_test_pred_classes = torch.argmax(y_test_pred, dim=1)
    y_test_true_classes = torch.argmax(y_test_tensor, dim=1)

    precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
    recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')
    f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')

    print(f'Precisión: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

# Función para guardar el modelo
def save_model(model, path='models/model_final.pth'):
    torch.save(model.state_dict(), path)
    print(f'Modelo guardado en {path}')
