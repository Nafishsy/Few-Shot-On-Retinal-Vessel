# Use the official Python image (change to a compatible version like 3.10)
FROM python:3.10-slim

# Install distutils
RUN apt-get update && apt-get install -y python3-distutils

# Set working directory
WORKDIR /app

# Install required dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Add a non-root user
RUN useradd -m myuser
USER myuser

# Copy app code
COPY . /app/

# Expose port 5000
EXPOSE 5000

# Set Flask to run in production mode
ENV FLASK_ENV=production

# Run the app using Python
CMD ["python", "api/index.py"]
