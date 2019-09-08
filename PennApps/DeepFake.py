
import os
import pafy
import youtube_dl
from flask import Flask, redirect, render_template, send_from_directory, request
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",  methods=['GET', 'POST'])
def upload():
    text = request.form['text']
    print(text)
   
    ex = "chmod 755 youtube-dl; ./youtube-dl "+text
    os.system(ex)
    return render_template('index.html', text=text)
	

if __name__ == '__main__': #true if you run the script directly
    app.run(debug=True)