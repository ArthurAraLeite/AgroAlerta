from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/dados")
def dados():
    return jsonify({
        "ph": 6.2,
        "umidade": 34,
        "nitrogenio": 120,
        "fosforo": 45,
        "potassio": 200,
        "condutividade": 1.8,
        "temperatura": 26.5,
        "penetrometro": 3.1
    })

if __name__ == "__main__":
    app.run(port=5000)