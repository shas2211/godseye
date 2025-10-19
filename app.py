from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import insightface
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load InsightFace model
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)  # CPU: -1, GPU: 0

enroll_embedding = None
threshold = 0.35  # ArcFace threshold

# -----------------------------
# Helper functions
# -----------------------------
def get_face_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    faces = model.get(img)
    if not faces:
        return None
    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def draw_cool_box(frame, bbox, sim, threshold):
    x1, y1, x2, y2 = bbox.astype(int)
    color = (255, 255, 255)  # white
    # Glow effect: multiple rectangles
    for i in range(3):
        cv2.rectangle(frame, (x1-i, y1-i), (x2+i, y2+i), color, 1)

    # Text overlay with black background
    text = f"Match {sim:.2f}" if sim > threshold else "Unknown"
    ((text_w, text_h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Similarity bar
    bar_length = int((x2-x1) * min(max(sim,0),1))
    cv2.rectangle(frame, (x1, y2+5), (x1+bar_length, y2+15), color, -1)

# -----------------------------
# Webcam generator
# -----------------------------
def gen_frames():
    global enroll_embedding
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        faces = model.get(frame)
        for face in faces:
            sim = cosine_similarity(enroll_embedding, face.embedding)
            draw_cool_box(frame, face.bbox, sim, threshold)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    global enroll_embedding
    if request.method == 'POST':
        if 'enroll_img' not in request.files:
            return "No file part"
        file = request.files['enroll_img']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        emb = get_face_embedding(filepath)
        if emb is None:
            return "No face detected in uploaded image"
        enroll_embedding = emb
        return render_template('index.html', cam_ready=True)
    return render_template('index.html', cam_ready=False)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
