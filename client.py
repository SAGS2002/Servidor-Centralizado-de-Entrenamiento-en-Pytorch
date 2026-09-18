import socket
import os
import json

def enviar_imagen(host, puerto, ruta_imagen, etiqueta):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, puerto))
        
        nombre_archivo = os.path.basename(ruta_imagen)
        tamano_archivo = os.path.getsize(ruta_imagen)
        
        metadatos = json.dumps({
            "etiqueta": etiqueta,
            "archivo": nombre_archivo,
            "tamano": tamano_archivo
        }).encode('utf-8')
        
        s.sendall(len(metadatos).to_bytes(4, byteorder='big'))
        s.sendall(metadatos)
        
        with open(ruta_imagen, "rb") as f:
            while chunk := f.read(4096):
                s.sendall(chunk)
                
        respuesta = s.recv(1024).decode('utf-8')
        print(f"Servidor: {respuesta}")

if __name__ == "__main__":
    print("--- Cliente de Envio de Imagenes ---")
    ruta = input("Ingresa la ruta de la imagen (ej. gato.png): ")
    etiqueta = input("Ingresa la etiqueta (ej. gato, perro): ")
    
    if os.path.exists(ruta):
        enviar_imagen('127.0.0.1', 5000, ruta, etiqueta)
    else:
        print("Error: El archivo no existe. Verifica la ruta.")