from flask import Flask, render_template, request, jsonify,send_from_directory
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    json_path = request.json.get('path')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# @app.route('/images/<path:filename>')
# def serve_image(filename):
#     directory = '/media/airs/BIN/cwah_ex/images'  # Update this path to your image directory
#     return send_from_directory(directory, filename)
@app.route('/media/<path:filename>')
def serve_image(filename):
    image_directory = '/media/'  # Update to your image directory
    return send_from_directory(image_directory,filename)

if __name__ == '__main__':
    app.run(debug=False)