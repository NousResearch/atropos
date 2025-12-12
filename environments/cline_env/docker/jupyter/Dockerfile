# Cline Jupyter Environment (Python with Jupyter)
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Python packages for Jupyter
RUN pip3 install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipython \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn

# Setup for cline user
RUN mkdir -p /home/cline/.jupyter /home/cline/.local \
    && chown -R cline:cline /home/cline/.jupyter /home/cline/.local

USER cline

# Verify installation
RUN jupyter --version

WORKDIR /workspace
