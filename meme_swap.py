import os

from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def meme_swap():
    return render_template("meme_swap.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        filename = request.form['file']

        print(filename)

        return jsonify(request.form['file'])

    return render_template("meme_snap.html")

if __name__ == '__main__':
    app.run(debug=True)
