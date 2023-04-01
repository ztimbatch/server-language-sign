from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def alive():
    return jsonify({'response': 'server sign language is alive'})


if __name__ == '__main__':
    app.run(debug=True)
