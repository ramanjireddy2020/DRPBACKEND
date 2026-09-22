FROM ubuntu:22.04

# Set non-interactive mode to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm miniconda.sh

# Set up Conda environment — use conda-forge only (removes defaults to avoid ToS)
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda config --remove-key channels && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority disabled && \
    conda install -c conda-forge openbabel && \
    conda clean --all -y

# Install AutoDock Vina
WORKDIR /opt
RUN wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina-linux_x86_64 -O /usr/local/bin/vina \
    && chmod +x /usr/local/bin/vina

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
