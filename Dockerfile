# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container to / (root directory)
# This is optional since the root is the default work directory, but it's a good practice to define it explicitly
WORKDIR /

# Copy the current directory contents into the container at /
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT=5000

# Use Gunicorn to serve the app
CMD gunicorn --bind 0.0.0.0:$PORT app:app
