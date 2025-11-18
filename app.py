import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    print("✅ Model TFLite berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    interpreter = None

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

FULL_CLASS_NAMES = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

RECOMMENDATIONS = {
    'nv': """
        <strong>Status:</strong> Umumnya jinak (tahi lalat biasa).
        <br><strong>Pencegahan:</strong> Lakukan pemeriksaan kulit mandiri secara rutin (metode ABCDE - Asymmetry, Border, Color, Diameter, Evolving). Lindungi kulit dari paparan sinar UV berlebih.
        <br><strong>Penanggulangan:</strong> Umumnya tidak berbahaya. Namun, jika Anda melihat ada perubahan pada bentuk, ukuran, warna, atau terasa gatal/berdarah, <strong>segera periksakan ke dokter</strong> untuk menyingkirkan kemungkinan melanoma.
        """,
    'mel': """
        <strong>Status:</strong> Berpotensi serius (kanker kulit melanoma).
        <br><strong>Pencegahan:</strong> Gunakan tabir surya setiap hari (SPF 30+), hindari jam puncak sinar matahari (10 pagi - 4 sore), dan kenakan pakaian pelindung.
        <br><strong>Penanggulangan:</strong> <strong>SANGAT PENTING untuk segera berkonsultasi dengan dokter kulit (dermatologis).</strong> Deteksi dini adalah kunci keberhasilan perawatan. Jangan menunda pemeriksaan.
        """,
    'bkl': """
        <strong>Status:</strong> Jinak (seperti keratosis seboroik).
        <br><strong>Pencegahan:</strong> Tidak sepenuhnya dapat dicegah, seringkali berkaitan dengan faktor genetik dan penuaan.
        <br><strong>Penanggulangan:</strong> Umumnya tidak memerlukan perawatan medis karena bersifat jinak. Jika lesi terasa gatal, iritasi, atau mengganggu secara kosmetik, dokter dapat menghilangkannya dengan prosedur sederhana.
        """,
    'bcc': """
        <strong>Status:</strong> Kanker kulit (karsinoma sel basal).
        <br><strong>Pencegahan:</strong> Perlindungan ketat dari sinar UV adalah kuncinya. Gunakan tabir surya dan hindari paparan matahari berlebih.
        <br><strong>Penanggulangan:</strong> <strong>Segera konsultasikan dengan dokter.</strong> BCC biasanya tumbuh lambat dan sangat dapat disembuhkan jika ditangani sejak dini, seringkali melalui pembedahan kecil atau perawatan topikal.
        """,
    'akiec': """
        <strong>Status:</strong> Pra-kanker (keratosis aktinik).
        <br><strong>Pencegahan:</strong> Disebabkan oleh kerusakan akibat sinar matahari. Perlindungan UV yang ketat (tabir surya, topi, pakaian pelindung) sangat penting.
        <br><strong>Penanggulangan:</strong> <strong>Penting untuk diperiksa oleh dokter.</strong> Lesi ini memiliki potensi untuk berkembang menjadi kanker kulit. Perawatan mungkin termasuk krioterapi (pembekuan), krim resep, atau terapi lainnya.
        """,
    'vasc': """
        <strong>Status:</strong> Umumnya jinak (lesi vaskular seperti angioma).
        <br><strong>Pencegahan:</strong> Umumnya tidak dapat dicegah, seringkali bersifat bawaan atau berkembang seiring bertambahnya usia.
        <br><strong>Penanggulangan:</strong> Biasanya tidak berbahaya dan tidak memerlukan perawatan. Jika mudah berdarah atau mengganggu secara kosmetik, dokter dapat menanganinya (misalnya dengan laser).
        """,
    'df': """
        <strong>Status:</strong> Jinak (dermatofibroma).
        <br><strong>Pencegahan:</strong> Tidak dapat dicegah. Sering muncul setelah cedera ringan pada kulit (seperti gigitan serangga).
        <br><strong>Penanggulangan:</strong> Tidak berbahaya dan tidak memerlukan perawatan. Lesi ini biasanya keras saat disentuh. <strong>Konsultasikan dengan dokter</strong> hanya untuk memastikan diagnosis yang benar.
        """,
    'default': """
        Rekomendasi tidak tersedia. Silakan konsultasikan dengan dokter kulit untuk informasi lebih lanjut.
        """
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_image(image_path):
    if interpreter is None:
        return None

    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    
    all_predictions = {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
    return all_predictions

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        all_predictions = predict_image(filepath)
        
        if all_predictions:
            top_label_short = max(all_predictions, key=all_predictions.get)
            top_label_full = FULL_CLASS_NAMES.get(top_label_short, "Tidak Diketahui")
            top_confidence = all_predictions[top_label_short]
            
            recommendation = RECOMMENDATIONS.get(top_label_short, RECOMMENDATIONS['default'])

        else:
            top_label_full = "Error"
            top_confidence = 0
            recommendation = RECOMMENDATIONS['default']
        
        return render_template(
            'index.html', 
            filename=filename, 
            top_label=top_label_full, 
            top_confidence=f"{top_confidence * 100:.2f}",
            recommendation=recommendation
        )

    return render_template('index.html', filename=None)

if __name__ == '__main__':
    app.run(debug=True)