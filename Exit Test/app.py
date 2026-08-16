from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('eurovision_model.joblib')
model_columns = joblib.load('model_columns.joblib')

def get_country_list():
    """Helper function to load all unique country names safely from the dataset."""
    try:
        xls = pd.ExcelFile('dataset.xlsx')
        voting_df = pd.read_excel(xls, sheet_name='Voting Final')
        voting_df.columns = voting_df.columns.str.strip()
        col_map = {c.lower(): c for c in voting_df.columns}
        country_col = col_map.get('country')
        if country_col:
            countries = voting_df[country_col].dropna().unique().tolist()
            return sorted(countries)
        return ["Sweden", "United Kingdom", "France", "Ukraine"]
    except Exception:
        return ["Sweden", "United Kingdom", "France", "Ukraine"]

@app.route('/')
def home():
    countries = get_country_list()
    return render_template('index.html', prediction_text='', voting_text='', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        song_in_english = request.form['Song_In_English']
        group_solo = request.form['Group_Solo']
        artist_gender = request.form['Artist_gender']
        
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        year = int(request.form['Year'])
        tempo = float(request.form['tempo'])
        loudness = float(request.form['loudness'])
        
        input_dict = {col: 0 for col in model_columns}
        
        defaults = {
            'duration': 180.0,
            'acousticness': 0.2,
            'speechiness': 0.05,
            'key': 5,
            'liveness': 0.15,
            'time_signature': 4,
            'mode': 1,
            'valence': 0.5,
            'Happiness': 0.5
        }
        for col, val in defaults.items():
            if col in input_dict:
                input_dict[col] = val

        if 'Year' in input_dict: input_dict['Year'] = year
        if 'danceability' in input_dict: input_dict['danceability'] = danceability
        if 'energy' in input_dict: input_dict['energy'] = energy
        if 'tempo' in input_dict: input_dict['tempo'] = tempo
        if 'loudness' in input_dict: input_dict['loudness'] = loudness
        
        if song_in_english == 'Yes' and 'Song.In.English_Yes' in input_dict:
            input_dict['Song.In.English_Yes'] = 1
            
        if group_solo == 'Solo' and 'Group.Solo_Solo' in input_dict:
            input_dict['Group.Solo_Solo'] = 1
            
        if artist_gender == 'Male' and 'Artist.gender_Male' in input_dict:
            input_dict['Artist.gender_Male'] = 1
        elif artist_gender == 'Female' and 'Artist.gender_Female' in input_dict:
            input_dict['Artist.gender_Female'] = 1

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)
        output = round(prediction[0])
        
        prediction_text = f"Predicted Eurovision Points: {output}"

    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    countries = get_country_list()
    return render_template('index.html', prediction_text=prediction_text, voting_text='', countries=countries)

@app.route('/voting_history', methods=['POST'])
def voting_history():
    try:
        selected_country = request.form['country']
        
        xls = pd.ExcelFile('dataset.xlsx')
        voting_df = pd.read_excel(xls, sheet_name='Voting Final')
        
        # Strip trailing/leading whitespaces from column headers
        voting_df.columns = voting_df.columns.str.strip()
        
        # Map columns case-insensitively and handle both 'points' or 'score'
        col_map = {c.lower(): c for c in voting_df.columns}
        country_col = col_map.get('country')
        giver_col = col_map.get('giver')
        points_col = col_map.get('points') or col_map.get('score')
        
        if not country_col or not giver_col or not points_col:
            voting_text = f"Error: Columns missing in 'Voting Final'. Found: {list(voting_df.columns)}"
        else:
            received_votes = voting_df[voting_df[country_col] == selected_country]
            best_friend_series = received_votes.groupby(giver_col)[points_col].sum().sort_values(ascending=False)
            
            if not best_friend_series.empty:
                best_friend = best_friend_series.index[0]
                max_points = best_friend_series.iloc[0]
                voting_text = f"Historically, {best_friend} is {selected_country}'s best friend, giving a total of {max_points} points!"
            else:
                voting_text = f"No historical voting data found for {selected_country}."

    except Exception as e:
        voting_text = f"An error occurred: {e}"

    countries = get_country_list()
    return render_template('index.html', prediction_text='', voting_text=voting_text, countries=countries)

if __name__ == "__main__":
    app.run(debug=True)