from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
import datetime
from datetime import timedelta
import pytz
import gmplot
import numpy
import pandas
import requests
import pickle
import sklearn, sklearn.tree, sklearn.ensemble, sklearn.feature_extraction, sklearn.metrics

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace 'haha' with a more secure secret key for production
app.permanent_session_lifetime = timedelta(minutes=5)  # Session lifetime set to 5 minutes

apikey = "AIzaSyAw_BHttzAxFdQ4Lg5UT7oYTjJ5LFON2Z8"

# Parts of the ensemble model
rf = pickle.load(open('model_RandomForest.pk1', 'rb'))
ml_reg = pickle.load(open('model_MultipleRegression.pk1', 'rb'))

# Substitute for empty data obtained from API
# in the order of: ign_date, latitude, longitude, minimum temperature, maximum temperature, average temperature, dew point, relative humidity, heat index, wind speed, wind gust, wind direction, wind chill, precipitation, visibility, cloud cover, sea level pressure
avg_for_na = [
    20026124.90970577, 
    53.40442305249716, 
    -122.14423555004001,
    40.58008083140884,
    63.09509237875295,
    52.084209006928425,
    37.187727006444,
    62.41874633860559,
    85.18183962264142,
    12.53463329452857,
    27.2919663351186,
    192.44023689197786,
    27.720631578947398,
    0.06612954545454518,
    76.20172189733584,
    30.337810514153652,
    1015.3026817640034
]

# Ensemble model definition
def ensemble_model(input_df):
    # Use DictVectorizer to transform the dataframe into a format suitable for the random forest model
    dvr = sklearn.feature_extraction.DictVectorizer()
    rf_input = dvr.fit_transform(input_df.T.to_dict().values())
    
    # Predict using the random forest model
    rf_pred = rf.predict(rf_input)
    
    # Predict using the multilinear regression model
    ml_reg_pred = ml_reg.predict(input_df)
    
    # Return the average of both model predictions
    return (rf_pred + ml_reg_pred) / 2.0


# Function to get tomorrow's date in British Columbia in a model-friendly format (YYYYMMDD)
def get_tommorow_date_BC():
    # Obtain the current date and time for BC, then add one day to get tomorrow's date
    my_date = datetime.datetime.now(pytz.timezone('US/Pacific')) + datetime.timedelta(days=1)
    
    # Format month and day to ensure two digits (e.g., 04 for April, 09 for 9th day of the month)
    proper_month = f'{my_date.month:02d}'
    proper_day = f'{my_date.day:02d}'
    
    # Construct the date string in YYYYMMDD format
    return int(f'{my_date.year}{proper_month}{proper_day}')



def split(word):
  return [char for char in word]

# Function to retrieve climate data from Visual Crossing Weather API for a given latitude and longitude
def get_future_climate_data(latitude, longitude):
    # Define the API endpoint
    url = "https://visual-crossing-weather.p.rapidapi.com/forecast"
    
    # Create the coordinates string from latitude and longitude
    coords = f'{latitude}, {longitude}'
    
    # Set up the parameters for the API request
    querystring = {
        "location": coords,
        "aggregateHours": "24",  # Aggregates the weather data into 24-hour blocks
        "shortColumnNames": "0",  # Use full column names in the output
        "unitGroup": "us",  # Uses US units (e.g., Fahrenheit for temperature)
        "contentType": "csv"  # Requests data in CSV format
    }
    
    # Specify the headers including the API key and host
    headers = {
        'x-rapidapi-key': "5416e60fd3msh25313be4cf91e39p1db2f2jsn7acc7269aacf",
        'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com"
    }
    
    # Make the GET request to the API
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    # Split the CSV formatted text by commas into a list
    response_list = response.text.split(",")

    return response_list

# Function to create a model-friendly input DataFrame from weather data and user inputs
def get_input_df(ign_date, latitude, longitude):
    # Helper function to convert empty or 'N/A' entries to NaN for better handling in pandas
    def to_nan(entry):
        return numpy.nan if entry in ('', 'N/A') else entry

    # Retrieve climate data from the API
    response_list = get_future_climate_data(latitude, longitude)
    
    # Ensure response has enough data, if not, fill with 'N/A'
    if len(response_list) < 52:  # Adjusted the check to match the expected number of data points
        response_list = ['N/A'] * 52
    
    # Map the response to the expected structure for the model, using predefined averages for missing data
    input_data = {
        'IGN_DATE': ign_date, 'LATITUDE': latitude, 'LONGITUDE': longitude,
        'minimum temperature': response_list[31], 'maximum temperature': response_list[32],
        'average temperature': response_list[33], 'dew point': avg_for_na[6], 
        'relative humidity': response_list[42], 'heat index': response_list[36], 
        'wind speed': response_list[34], 'wind gust': response_list[43],
        'wind direction': response_list[30], 'wind chill': response_list[44], 
        'precipitation': response_list[38], 'visibility': avg_for_na[14], 
        'cloud cover': response_list[35], 'sea level pressure': response_list[39]
    }
    
    # Create a DataFrame from the input data
    input_data_frame = pandas.DataFrame(data=input_data, index=[0])
    print(input_data_frame)
    
    # Replace 'N/A' or empty strings with NaN and fill them with predefined averages
    for count, column in enumerate(input_data_frame.columns):
        input_data_frame[column] = input_data_frame[column].apply(to_nan)
        input_data_frame[column].fillna(avg_for_na[count], inplace=True)

    return input_data_frame

# Route for the home page
@app.route('/', methods=['POST', 'GET'])
def home_page():
    # Handle form submission
    if request.method == 'POST':
        # Retrieve coordinates from the form
        longitude_cord = request.form['longitude_textbar']
        latitude_cord = request.form['latitude_textbar']

        # Validate if the coordinates are within the defined range for British Columbia
        if -130.0 <= float(longitude_cord) <= -115.0 and 48.0 <= float(latitude_cord) <= 60.0:
            # Redirect to the map page if valid
            return redirect(url_for("map_page", longitude=longitude_cord, latitude=latitude_cord))
        else:
            # Show an error message if coordinates are not valid
            flash("Error: Coordinates do not point to a location in/near to BC, try again", "info")
            return redirect(url_for('home_page'))
    else:
        # Render the home page where users can input coordinates
        return render_template('webpage.html')

# Route to display the results on a map after valid coordinates are submitted
@app.route('/result')

def map_page():
    # Get tomorrow's date in BC in a suitable format for the model
    tomorrow_date_BC = get_tommorow_date_BC()
    # Retrieve coordinates from request parameters
    longitude = float(request.args.get('longitude'))
    latitude = float(request.args.get('latitude'))
    # Generate lists of coordinates to check around the central point
    longitudes_within_range = [longitude - 0.1, longitude, longitude + 0.1]
    latitudes_within_range = [latitude - 0.1, latitude, latitude + 0.1]
    # Define the threshold for considering a fire risk as significant
    threshold_fire = numpy.log(3)
    lower_than_threshold = []
    higher_than_threshold = []
    size_of_fire = []

    # Loop through the generated lists of longitudes and latitudes
    for long in longitudes_within_range:
      for lat in latitudes_within_range:
        # Get the DataFrame for model prediction
        input_for_model = get_input_df(tomorrow_date_BC, lat, long)
        # Predict fire risk using the ensemble model
        prediction = ensemble_model(input_for_model)

        # Ensure prediction is a scalar by converting it if necessary
        if isinstance(prediction, numpy.ndarray):
            prediction = prediction.item()  # Convert single-element array to scalar

        print(f"Prediction for {lat}, {long}: {prediction}")  # Debugging: print the prediction value

        # Categorize the location based on the prediction threshold
        if prediction > threshold_fire:
            higher_than_threshold.append((lat, long))
            size_of_fire.append(prediction)
        else:
            lower_than_threshold.append((lat, long))

    # Initialize a Google map centered at the given coordinates
    gmap = gmplot.GoogleMapPlotter(latitude, longitude, 10, apikey=apikey)

    # Add markers for points with low fire risk
    if len(lower_than_threshold) > 0:
        no_fire_lats, no_fire_lngs = zip(*lower_than_threshold)
        gmap.scatter(no_fire_lats, no_fire_lngs, color='#808080', size=1000, marker=False)

    # Add markers for points with high fire risk and label them with estimated fire size
    if len(higher_than_threshold) > 0:
        for idx, (lat, long) in enumerate(higher_than_threshold):
            # Calculate the estimated size of the fire
            fire_size = numpy.exp(size_of_fire[idx]) - 1
            # Format the title string with fire size information
            title = f"Approx-size: {fire_size:.2f} Hectares"
            gmap.marker(lat, long, title=title)

    # Draw an area representing the range of the map
    area = [(latitude + dlat, longitude + dlong) for dlat in [-0.1, 0.1, 0.1, -0.1, -0.1] for dlong in [-0.1, 0.1]]
    gmap.plot(*zip(*area), color='cornflowerblue', edge_width=1)

    # Generate the map HTML and save it to the templates directory
    gmap.draw('templates/map.html')
    # Serve the map HTML to the user
    return render_template('map.html')
    
#if __name__ == "__main__":
    #app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
