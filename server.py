import socket
import threading
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AnimalCNN

HOST = '127.0.0.1'
PORT = 5000
DATASET_DIR = 'dataset'
BATCH_SIZE_ENTRENAMIENTO = 4 

nuevas_imagenes = 0
lock = threading.Lock()

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = AnimalCNN(num_classes=2).to(dispositivo) 
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

def manejar_cliente(conn, addr):
    global nuevas_imagenes
    try:
        meta_len = int.from_bytes(conn.recv(4), byteorder='big')
        metadatos = json.loads(conn.recv(meta_len).decode('utf-8'))
        
        etiqueta = metadatos['etiqueta']
        tamano = metadatos['tamano']
        archivo = metadatos['archivo']
        
        directorio_clase = os.path.join(DATASET_DIR, etiqueta)
        os.makedirs(directorio_clase, exist_ok=True)
        
        ruta_guardado = os.path.join(directorio_clase, archivo)
        
        recibido = 0
        with open(ruta_guardado, 'wb') as f:
            while recibido < tamano:
                chunk = conn.recv(min(4096, tamano - recibido))
                if not chunk:
                    break
                f.write(chunk)
                recibido += len(chunk)
        
        conn.sendall(b"Imagen recibida y aceptada para entrenamiento")
        
        with lock:
            nuevas_imagenes += 1
            
    except Exception as e:
        print(f"Error con el cliente {addr}: {e}")
    finally:
        conn.close()

def bucle_entrenamiento():
    global nuevas_imagenes
    
    transformacion = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    EPOCAS = 10 # Número de veces que la red repasará las imágenes
    
    while True:
        time.sleep(5)
        
        iniciar = False
        with lock:
            if nuevas_imagenes >= BATCH_SIZE_ENTRENAMIENTO:
                iniciar = True
                nuevas_imagenes = 0
                
        if iniciar:
            print(f"\n[Entrenamiento] Iniciando ciclo de {EPOCAS} epocas...")
            try:
                dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transformacion)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_ENTRENAMIENTO, shuffle=True)
                
                modelo.train()
                
                # Nuevo bucle: Repetir el estudio múltiples veces
                for epoca in range(EPOCAS):
                    loss_acumulada = 0.0
                    correctos = 0
                    total = 0
                    
                    for inputs, labels in dataloader:
                        inputs, labels = inputs.to(dispositivo), labels.to(dispositivo)
                        
                        optimizador.zero_grad()
                        outputs = modelo(inputs)
                        loss = criterio(outputs, labels)
                        loss.backward()
                        optimizador.step()
                        
                        loss_acumulada += loss.item()
                        _, predichos = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correctos += (predichos == labels).sum().item()
                    
                    accuracy = 100 * correctos / total
                    print(f"  -> Epoca [{epoca+1}/{EPOCAS}] | Loss: {loss_acumulada/len(dataloader):.4f} | Precision: {accuracy:.2f}%")
                
                torch.save(modelo.state_dict(), "modelo_entrenado.pth")
                print("[Guardado] Cerebro actualizado y guardado en modelo_entrenado.pth")
                
            except Exception as e:
                print(f"[Entrenamiento] Esperando mas clases o imagenes validas... ({e})")


def iniciar_servidor():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    hilo_entrenamiento = threading.Thread(target=bucle_entrenamiento, daemon=True)
    hilo_entrenamiento.start()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Servidor de PyTorch escuchando en {HOST}:{PORT}...")
        
        while True:
            conn, addr = s.accept()
            threading.Thread(target=manejar_cliente, args=(conn, addr)).start()

if __name__ == "__main__":
    iniciar_servidor()