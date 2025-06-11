#!/bin/bash
# NDML Installation Script
# Installs all dependencies and sets up the environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== NDML Installation Script ===${NC}"

# Check system requirements
check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"

    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ ! "$python_version" =~ ^3\.(8|9|10|11) ]]; then
        echo -e "${RED}Error: Python 3.8+ required. Found: $python_version${NC}"
        exit 1
    fi

    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}CUDA available${NC}"
        cuda_available=true
    else
        echo -e "${YELLOW}Warning: CUDA not detected. GPU features will be disabled.${NC}"
        cuda_available=false
    fi

    # Check RAM
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 32 ]; then
        echo -e "${YELLOW}Warning: Less than 32GB RAM detected. Performance may be impacted.${NC}"
    fi
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"

    # Create virtual environment
    python3 -m venv ndml_env
    source ndml_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install PyTorch with CUDA support
    if [ "$cuda_available" = true ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install core dependencies
    pip install -r requirements.txt

    # Install the package in editable mode
    pip install -e .

    # Install FAISS
    if [ "$cuda_available" = true ]; then
        pip install faiss-gpu
    else
        pip install faiss-cpu
    fi
}

# Setup directory structure
setup_directories() {
    echo -e "${YELLOW}Setting up directory structure...${NC}"

    directories=(
        "/opt/ndml/config"
        "/opt/ndml/data/memory"
        "/opt/ndml/data/checkpoints"
        "/opt/ndml/logs"
        "/opt/ndml/models"
        "/opt/ndml/monitoring"
        "/opt/ndml/scripts"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        echo "Created: $dir"
    done

    # Set permissions
    chmod -R 755 /opt/ndml
}

# Install Redis
install_redis() {
    echo -e "${YELLOW}Installing Redis...${NC}"

    if command -v redis-server &> /dev/null; then
        echo "Redis already installed"
    else
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y redis-server
            sudo systemctl enable redis-server
            sudo systemctl start redis-server
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install redis
            brew services start redis
        fi
    fi
}

# Download pre-trained models
download_models() {
    echo -e "${YELLOW}Downloading pre-trained models...${NC}"

    # Create models directory
    mkdir -p /opt/ndml/models

    # Download models based on configuration
    python3 scripts/download_models.py --config config/models.yaml
}

# Configure environment variables
configure_environment() {
    echo -e "${YELLOW}Configuring environment...${NC}"

    cat > /opt/ndml/.env << EOF
# NDML Environment Configuration
export NDML_HOME=/opt/ndml
export NDML_DATA_PATH=/opt/ndml/data
export NDML_CONFIG_PATH=/opt/ndml/config
export NDML_LOG_PATH=/opt/ndml/logs
export NDML_MODEL_PATH=/opt/ndml/models
export NDML_LOG_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Redis Configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Add NDML to Python path
export PYTHONPATH=\$PYTHONPATH:/opt/ndml
EOF

    # Source environment
    echo "source /opt/ndml/.env" >> ~/.bashrc
}

# Run installation
main() {
    check_requirements
    install_python_deps
    setup_directories
    install_redis
    configure_environment

    echo -e "${GREEN}=== NDML Installation Complete ===${NC}"
    echo -e "${YELLOW}Please run: source ~/.bashrc${NC}"
    echo -e "${YELLOW}Then activate virtual environment: source ndml_env/bin/activate${NC}"
}

main