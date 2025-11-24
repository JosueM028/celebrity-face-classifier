import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --- CONFIGURACIÓN ---
app = Flask(__name__)

# Rutas
# Subimos un nivel (..) para buscar la carpeta models
MODEL_PATH = '../models/celebrity_vgg16.keras'
UPLOAD_FOLDER = 'static/uploads'

# Crear carpeta de subidas si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- CARGAR EL MODELO ---
print("Cargando modelo VGG16... Espere un momento.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    print("Asegúrate de haber movido el archivo .keras a la carpeta 'models'")

# --- LISTA DE CLASES (Orden Alfabético EXACTO) ---
CLASS_NAMES = [
    "Denzel Washington",
    "Hugh Jackman",
    "Jennifer Lawrence",
    "Megan Fox",
    "Natalie Portman",
    "Scarlett Johansson"
]

def predict_label(img_path):
    try:
        # 1. Cargar imagen y redimensionar a 224x224 (Input de VGG16)
        # Usamos tf.keras.utils para evitar errores de importación
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        
        # 2. Convertir a Array
        img_array = tf.keras.utils.img_to_array(img)
        
        # 3. Expandir dimensiones (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 4. Predecir
        prediction = model.predict(img_batch)
        
        # 5. Obtener resultados
        index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        return CLASS_NAMES[index], confidence
    except Exception as e:
        print(f"Error en predicción: {e}")
        return "Error", 0.0

# --- RUTAS DE LA WEB ---
@app.route("/", methods=['GET', 'POST'])
def main():
    prediction = None
    img_path = None
    confidence_str = None
    confidence_val = 0 # Valor numérico para la lógica del HTML

    if request.method == 'POST':
        img = request.files['my_image']
        if img:
            filename = secure_filename(img.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)
            
            # Llamamos a la función de predicción
            predicted_class, conf = predict_label(img_path)
            
            prediction = predicted_class
            confidence_val = conf
            confidence_str = f"{conf:.2f}%"

    # Renderizar plantilla con todas las variables
    return render_template(
        "index.html", 
        prediction=prediction, 
        confidence=confidence_str,
        confidence_val=confidence_val,
        img_path=img_path
    )

if __name__ =='__main__':
    app.run(debug=True)