# Use the official Ubuntu base image
FROM python:3.8-slim-buster

# Set the working directory inside the container
WORKDIR /app/

# Update the package lists and install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements file into the container at /app
COPY setup.py /app/
COPY init_setup.sh /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY files /app/files/
COPY static /app/static/
COPY sample_image_inputs_to_app  /app/sample_image_inputs_to_app/
COPY components/ /app/components/
COPY app.py /app/
COPY templates /app/templates/
COPY test_results /app/test_results/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "app.py"]
# CMD ["ls"]