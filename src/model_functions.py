import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


# Función para cargar los datos
def load_data(csv_path='data/raw/features_30_sec.csv'):
    """Cargar y preprocesar los datos del archivo CSV"""
    data = pd.read_csv(csv_path)
    X = data.drop(columns=['filename', 'length', 'label'])
    y = data['label']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_one_hot = np.eye(10)[y]  # Convertir las etiquetas a one-hot encoding
    return X, y_one_hot

# Función para dividir los datos
def split_data(X, y, test_size=0.3, val_size=0.5):
    """Dividir los datos en entrenamiento, validación y prueba"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para convertir DataFrame o numpy array a tensor
def convert_to_tensor(X, y):
    """
    Convierte un DataFrame o numpy array en tensores de PyTorch.
    Argumentos:
    - X: DataFrame o numpy array con las características (features).
    - y: numpy array o one-hot encoding con las etiquetas (labels).

    Retorna:
    - X_tensor: Tensor de PyTorch para las características.
    - y_tensor: Tensor de PyTorch para las etiquetas.
    """
    X_tensor = torch.FloatTensor(X.values)  # Convertir DataFrame a tensor
    y_tensor = torch.FloatTensor(y)         # Convertir las etiquetas a tensor
    return X_tensor, y_tensor

# Función para obtener la función de activación deseada
def get_activation_function(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Función de activación '{name}' no es válida")

# Función para obtener la función de pérdida (loss)
def get_loss_function(name):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Función de pérdida '{name}' no es válida")

def build_model(config):
    """Construir un modelo flexible con varias capas y funciones de activación"""
    layers = []
    
    # Extraer los valores del diccionario de configuración
    input_size = config['input_size']           # Número de entradas (características)
    hidden_layers = config.get('hidden_layers', [])  # Capas ocultas, puede ser vacío
    output_size = config['output_size']         # Número de salidas
    activation_fn = get_activation_function(config['activation_function'])  # Función de activación
    
    # Añadir las capas ocultas
    prev_size = input_size
    for hidden_size in hidden_layers:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(activation_fn)
        prev_size = hidden_size
    
    # Añadir la capa de salida
    layers.append(nn.Linear(prev_size, output_size))
    
    # Verificar si se requiere alguna activación en la capa de salida
    if config['loss_function'] == 'MSELoss':
        layers.append(nn.Sigmoid())  # MSELoss requiere Sigmoid para obtener probabilidades
    elif config['output_activation'] != 'none':
        layers.append(get_activation_function(config['output_activation']))  # Para otros casos

    # Definir el modelo secuencialmente
    return nn.Sequential(*layers)

# Modificar la función de entrenamiento para almacenar y mostrar las pérdidas solo del entrenamiento
def train_model(model, X_train_tensor, y_train_tensor, config):
    """Entrenar el modelo y visualizar las pérdidas al final del entrenamiento"""
    criterion = get_loss_function(config['loss_function'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate']) if config['optimizer'] == 'SGD' else torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    epochs = config['epochs']
    
    # Inicializar lista para almacenar las pérdidas de entrenamiento
    loss_train_history = []

    for epoch in range(epochs):
        # Forward pass en entrenamiento
        y_train_pred = model(X_train_tensor)
        loss_train = criterion(y_train_pred, y_train_tensor)
        
        # Reiniciar gradientes
        optimizer.zero_grad()
        
        # Backward pass y optimización
        loss_train.backward()
        optimizer.step()
        
        # Almacenar la pérdida de entrenamiento
        loss_train_history.append(loss_train.item())
        
        # Mostrar el progreso cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss Entrenamiento: {loss_train.item():.4f}')

    # Graficar las pérdidas completas al final del entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(loss_train_history, label='Loss Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Loss)')
    plt.title('Pérdida durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model




# Función para evaluar el modelo
def evaluate_model(model, X_test_tensor, y_test_tensor, config):
    """Evaluar el modelo en el conjunto de prueba"""
    criterion = get_loss_function(config['loss_function'])
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        loss_test = criterion(y_test_pred, y_test_tensor)
        print(f'Loss en el conjunto de prueba: {loss_test.item():.4f}')

# Función para guardar el modelo
def save_model(model, path='models/model_final.pth'):
    """Guardar el modelo entrenado en un archivo"""
    torch.save(model.state_dict(), path)
    print(f'Modelo guardado en {path}')