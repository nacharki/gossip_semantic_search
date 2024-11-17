# ------------------
# Stage 1: Builder
# ------------------
FROM condaforge/mambaforge:23.3.1-0 AS builder

# condaforge uses conda run as base, this leads to challenges with
# apt-get, so switch to use bash as the default RUN command
SHELL ["/bin/bash", "-c"]

# Set Conda environment variables and PATH
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Set environment variable to use the ADC credentials
COPY application_default_credentials.json /root/.config/gcloud/application_default_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json

# Install system dependencies and Python packages using mamba and pip
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

# Install system dependencies and Python packages using pip
RUN conda run -n base pip install -r /tmp/requirements.txt

# ------------------
# Stage 2: Runtime
# ------------------
FROM python:3.10-slim

# Set environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV AWS_PATH=/root/.local/bin
ENV PATH=$CONDA_DIR/bin:$AWS_PATH:$PATH

# Install git
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Copy essential files/folders from the builder stage to minimize image size
COPY --from=builder $CONDA_DIR $CONDA_DIR
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.config/gcloud/application_default_credentials.json /root/.config/gcloud/application_default_credentials.json

# Efficiently copy project code and other necessary files
COPY backend /workspaces/campaign/Research
COPY frontend /workspaces/campaign/campai


# Set the default command when the container starts (activate Conda environment)
SHELL ["conda", "run", "-n", "base", "/bin/bash"]


# Set the entrypoint to run the application
ENTRYPOINT ["python", "frontend/streamlit_app.py"]