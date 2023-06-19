from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socket = SocketIO(app)


@socket.on("model_loaded")
def model_loaded():
    print("Model loaded !")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    socket.run(app, port=5000, debug=True)
