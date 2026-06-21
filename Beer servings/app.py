import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and preprocessors
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_country = pickle.load(open('le_country.pkl', 'rb'))
le_continent = pickle.load(open('le_continent.pkl', 'rb'))

# Load dataset for infographics
df = pd.read_csv('beer-servings.csv') 

@app.route('/')
def home():
    # 1. Create the plot using Seaborn
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='beer_servings', y='total_litres_of_pure_alcohol', hue='continent')
    plt.title("Global Alcohol Consumption: Beer vs Total Pure Alcohol")
    
    # 2. Save plot to a memory buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close() # CRITICAL: Free memory
    
    # 3. Encode to base64 string
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('home.html', 
                           plot_url=plot_url,
                           countries=le_country.classes_, 
                           continents=le_continent.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the HTML form
        country_name = request.form['country']
        continent_name = request.form['continent']
        beer = float(request.form['beer'])
        spirit = float(request.form['spirit'])
        wine = float(request.form['wine'])

        # Apply Preprocessors
        country_val = le_country.transform([country_name])[0]
        continent_val = le_continent.transform([continent_name])[0]
        scaled_nums = scaler.transform([[beer, spirit, wine]])
        
        # Construct final input array
        features = np.array([country_val,scaled_nums[0][0],scaled_nums[0][1],scaled_nums[0][2],
        continent_val]).reshape(1, -1)
        
        # Predict
        output = model.predict(features)[0]
        output = round(output, 2)
        
        return render_template('res.html', 
                               prediction_text="Predicted Total Litres of Alcohol: {}".format(output))
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(port=8000)
