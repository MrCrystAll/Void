from flask import Flask, render_template
from flask_socketio import SocketIO
from multiprocessing import Process

from worker import Worker

app = Flask(__name__)
app.secret_key = "My secret key to not deliver".encode()
socket = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")



def rewards_changed(player, rewards):
    socket.emit("reward_change", {
        "rewards": rewards,
        "player": player
    }, namespace="/")


p = Process(target=Worker(rewards_changed).run, name="Worker")


if __name__ == "__main__":

    p.start()
    socket.run(app, debug=True, use_reloader=False)
