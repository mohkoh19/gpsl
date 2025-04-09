# Use a deep learning base image (PyTorch example)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Uninstall Conda to prevent conflicts
RUN rm -rf /opt/conda

# Install necessary system dependencies, including Python3-pip
RUN apt-get update && apt-get install -y git wget unzip python3-pip && ln -s /usr/bin/python3 /usr/bin/python && rm -rf /var/lib/apt/lists/*

# Create a working directory for your project
WORKDIR /app

# Install Poetry using pip
RUN pip install --no-cache-dir poetry

# Disable Poetry virtualenv creation (use system Python)
RUN poetry config virtualenvs.create false

# Copy dependency files first (to leverage Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of the code
COPY . .

# ================================
# PERSONAL ZSH CONFIGURATION
# This section is a personal preference
# Can be removed/commented out if not needed.
# Requires .zshrc and .p10k.zsh files in the same directory as the Dockerfile
# =================================

# Install Zsh and make it the default shell
RUN apt-get update && apt-get install -y zsh && chsh -s $(which zsh)

# Install Oh My Zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended

# Install Powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k

# Copy your custom .zshrc file into the container
COPY .zshrc /root/.zshrc

# Copy your custom .p10k.zsh file into the container
COPY .p10k.zsh /root/.p10k.zsh

# Disable Powerlevel10k configuration wizard
RUN echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >>! ~/.zshrc

# Add the app directory to the safe directory list
RUN zsh -c 'source ~/.zshrc && git config --global --add safe.directory /app'
