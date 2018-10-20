from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def meme_swap():
    return render_template("meme_swap.html")

@app.route('/', methods=['POST'])
def meme_swap_post():
    pass

if __name__ == '__main__':
    app.run()
