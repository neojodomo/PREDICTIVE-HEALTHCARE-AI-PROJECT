from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import logging
from datetime import datetime

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- APP SETUP ----------------
app = Flask(__name__)

# ---------------- DATABASE CONFIG ----------------
# This creates novacare.db automatically
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///novacare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- MODELS ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    name = db.Column(db.String(80), nullable=False)

class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    risk_score = db.Column(db.Float)
    glucose = db.Column(db.Integer)
# ---------------- LOAD MODEL ----------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.warning("Model not found. Using dummy prediction logic.")
    model = None


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "NovaCare Backend is running"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        logger.warning("Using dummy prediction logic.")

    data = request.get_json()

    try:
        features = [
            float(data.get("age", 0)),
            float(data.get("bp", 0)),
            float(data.get("glucose", 0))
        ]
    except Exception as e:
        logger.error(f"Invalid input data: {e}")
        return jsonify({"error": "Invalid input data"}), 400

    # Dummy fallback prediction
    if model:
        prediction = model(features)
    else:
        prediction = sum(features) / len(features)

    logger.info(
        f"Prediction made | Features: {features} | Risk Score: {prediction}"
    )

    return jsonify({"risk_score": prediction})

    # Dummy fallback if real model is not loaded
    if model:
        prediction = model(features)
    else:
        prediction = sum(features) / len(features)

    logger.info(f"Prediction made | Features: {features} | Risk Score: {prediction}")

    return jsonify({"risk_score": prediction})

# ---------------- CLI COMMAND: SEED DB ----------------
@app.cli.command("seed_db")
def seed_db():
    """Seeds the database with an admin user."""
    db.create_all()

    if not User.query.filter_by(email='admin@novacare.com').first():
        admin = User(
            email='admin@novacare.com',
            name='Admin User',
            password='admin123'  # NOTE: Hash in real apps
        )
        db.session.add(admin)
        db.session.commit()
        print(" Database seeded! Admin user created.")
    else:
        print(" Admin user already exists.")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
