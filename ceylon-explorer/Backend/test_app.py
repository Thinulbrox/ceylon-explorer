from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "🌴 Ceylon Explorer is working!"

@app.route('/test')
def test():
    return {"status": "Flask is working perfectly!"}

if __name__ == '__main__':
    print("Testing Flask installation...")
    app.run(debug=True, port=5000)