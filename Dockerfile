# Use the official Python image from Docker Hub
FROM python:3.12-slim

# Set a working directory for your app
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip

# Install required Python packages from requirements.txt without cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Add a non-root user to run the app for better security
RUN useradd -m myuser
USER myuser

# Copy the rest of the application code to the container
COPY . /app/

# Expose port 5000 for the Flask application (default port for Flask)
EXPOSE 5000

# Set Flask to run in production mode (optional, for security)
ENV FLASK_ENV=production

# Run the application using Python
CMD ["python", "app.py"]
