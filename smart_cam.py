from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('offer')
def handle_offer(offer):
    # 클라이언트로부터 받은 WebRTC offer 처리
    emit('offer', offer, broadcast=True)

@socketio.on('answer')
def handle_answer(answer):
    # WebRTC answer 처리
    emit('answer', answer, broadcast=True)

@socketio.on('ice-candidate')
def handle_ice_candidate(candidate):
    # ICE 후보 처리
    emit('ice-candidate', candidate, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
