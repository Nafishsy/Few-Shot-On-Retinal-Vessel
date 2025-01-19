FROM python:3.12-slim

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Copy your app
COPY . /app/

# Expose the port Vercel uses for the app
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
