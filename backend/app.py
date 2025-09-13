from flask import Flask, request, jsonify
import io
from PIL import Image
from enum import Enum

class CarState(Enum):
    Clean = 0
    Dirty = 1


app = Flask(__name__)

def evaluateCar(image: Image.Image) -> CarState:
    return CarState.Clean # TODO needs to be implemented by ML guys


@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    file = request.files['carImage'].read()
    image = Image.open(io.BytesIO(file))
    return jsonify(evaluateCar(image))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
