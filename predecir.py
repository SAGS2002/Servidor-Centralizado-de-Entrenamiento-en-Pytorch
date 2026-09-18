import torch
import os
from torchvision import transforms
from PIL import Image
from model import AnimalCNN

def predecir_imagen(ruta_imagen):
    clases = sorted(os.listdir('dataset'))
    num_classes = len(clases)
    
    modelo = AnimalCNN(num_classes=num_classes)
    modelo.load_state_dict(torch.load('modelo_entrenado.pth', map_location='cpu', weights_only=True))
    modelo.eval()
    
    transformacion = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen_tensor = transformacion(imagen).unsqueeze(0)
    
    with torch.no_grad():
        salida = modelo(imagen_tensor)
        _, prediccion_idx = torch.max(salida, 1)
        
        clase_predicha = clases[prediccion_idx.item()]
        
        print("\n" + "="*40)
        print(f"🤖 La red neuronal dice que esto es un: **{clase_predicha.upper()}**")
        print("="*40 + "\n")

if __name__ == "__main__":
    print("--- Sistema de Prediccion ---")
    ruta = input("Ingresa la ruta de la imagen a predecir: ")
    
    if not os.path.exists('modelo_entrenado.pth'):
        print("Error: El modelo no existe. ¡Debes enviar al menos 4 imagenes al servidor para que entrene y lo cree!")
    elif not os.path.exists(ruta):
        print("Error: La imagen no existe. Verifica la ruta.")
    else:
        predecir_imagen(ruta)