from flask import *
import os
from werkzeug.utils import secure_filename
import label_image

def load_image(image):
    text = label_image.main(image)
    return text

app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

 
  
    
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = load_image(file_path)
        result = result.title()
        d = {"Vitamin A":" → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
	'Vitamin B':" → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
        'Vitamin C':" → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
        'Vitamin D':" → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
        "Vitamin E":" → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."}
        result = result+d[result]
        #result2 = result+d[result]
        #result = [result]
        #result3 = d[result]        
        print(result)
        #print(result3)
        os.remove(file_path)
        return result
        #return result3
    return None

if __name__ == '__main__':
    app.run()