from io import BytesIO

from flask import Flask, request, Response, render_template, stream_with_context
from pydub import AudioSegment
from inference import inference
from flask_limiter import Limiter, util

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=util.get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
def get_midi_from_midi(data):
    save_path = BytesIO(data)
    save_path_with_left_right = BytesIO()
    inference(
        "example/checkpoint.pth",
        save_path,
        save_path_with_left_right
    )
    save_path_with_left_right.seek(0)
    
    return save_path_with_left_right

@limiter.limit("100/day;20/hour;2/minute")
@app.route('/')
def load_home():
    return render_template('index.html')

@limiter.limit("1000/day;200/hour;2/minute")
@app.route("/midi-to-midi", methods=["POST"])
def midi_to_midi():
    in_memory_file = BytesIO()
    chunk_size = 4 * 1024
    while True:
        chunk = request.stream.read(chunk_size)
        if len(chunk) == 0:
            break
        in_memory_file.write(bytes(chunk))

    in_memory_file.seek(0)
    midi_with_adjustments = get_midi_from_midi(in_memory_file.read())

    def generate():
        midi_with_adjustments.seek(0)
        chunk_size = 4 * 1024

        while True: #loop until the chunk is empty (the file is exhausted)
            chunk = midi_with_adjustments.read(chunk_size)
            yield chunk
            if (not chunk):
                break
        midi_with_adjustments.close()

    return Response(stream_with_context(generate()), mimetype='application/octet-stream')

if __name__ == "__main__":
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    app.run(
        host="0.0.0.0",
        port=5000
    )
