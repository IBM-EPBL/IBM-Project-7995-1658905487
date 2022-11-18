app = Flask(__name__,template_folder="../templates", 
            static_folder='../static')

model = load_model('nutrition.hdf5.h5')
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def image1():
    return render_template("image.html")

@app.route('/imageprediction')
def imageprediction():
    return render_template("imageprediction.html")