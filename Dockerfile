# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /workspace

# curl is used by the container healthcheck below.
# No build toolchain is needed: numpy/pandas/scikit-learn/scipy install from
# prebuilt manylinux wheels, so pip never compiles from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository contents into workspace directory
COPY . .

# Expose port 8501 for Streamlit App access
EXPOSE 8501

# Healthcheck to verify the web service is responding
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the streamlit application on startup
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
