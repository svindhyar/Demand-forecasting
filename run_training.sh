#!/bin/bash
# ==============================================================================
# Demand Forecasting - Setup and Training Script
# ==============================================================================

echo "=============================================================================="
echo "DEMAND FORECASTING PROJECT - SETUP & TRAINING"
echo "=============================================================================="
echo ""

# Navigate to project directory
cd /Users/Desktop/uniproject/retail_data

# ==============================================================================
# STEP 1: Environment Setup
# ==============================================================================

echo "STEP 1: Environment Setup"
echo "------------------------------------------------------------------------------"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
    
    # Create conda environment if it doesn't exist
    if conda env list | grep -q "demand_forecasting"; then
        echo "✓ Environment 'demand_forecasting' already exists"
    else
        echo "Creating conda environment from environment.yml..."
        conda env create -f environment.yml
        echo "✓ Environment created"
    fi
    
    # Activate environment
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate demand_forecasting
    echo "✓ Environment activated: demand_forecasting"
    
else
    echo "✓ Conda not found, using pip"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✓ Virtual environment created"
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

echo ""

# ==============================================================================
# STEP 2: Verify Installation
# ==============================================================================

echo "STEP 2: Verify Installation"
echo "------------------------------------------------------------------------------"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'✓ Scikit-learn: {sklearn.__version__}')"
python -c "import xgboost; print(f'✓ XGBoost: {xgboost.__version__}')"
python -c "import lightgbm; print(f'✓ LightGBM: {lightgbm.__version__}')"
echo ""

# ==============================================================================
# STEP 3: System Information
# ==============================================================================

echo "STEP 3: System Information"
echo "------------------------------------------------------------------------------"
echo "Python Version: $(python --version)"
echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory: $(sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}')"
echo "CUDA: Not applicable (CPU-based training on Mac)"
echo ""

# ==============================================================================
# STEP 4: Create Output Directories
# ==============================================================================

echo "STEP 4: Create Output Directories"
echo "------------------------------------------------------------------------------"
mkdir -p logs models results
echo "✓ Created: logs/, models/, results/"
echo ""

# ==============================================================================
# STEP 5: Training Options
# ==============================================================================

echo "STEP 5: Select Training Option"
echo "------------------------------------------------------------------------------"
echo "1. Basic Training (train_model.py)"
echo "2. Advanced Training - XGBoost (lr=0.05)"
echo "3. Advanced Training - LightGBM (lr=0.05)"
echo "4. Advanced Training - Both Models (production)"
echo "5. Skip training"
echo ""
read -p "Enter option (1-5): " option

case $option in
    1)
        echo ""
        echo "=============================================================================="
        echo "RUNNING: Basic Training"
        echo "Command: python train_model.py"
        echo "=============================================================================="
        python train_model.py
        ;;
    2)
        echo ""
        echo "=============================================================================="
        echo "RUNNING: Advanced Training - XGBoost"
        echo "Command: python train_model_advanced.py --model xgboost --xgb_lr 0.05 --xgb_max_depth 10 --xgb_n_estimators 300 --save_models"
        echo "=============================================================================="
        python train_model_advanced.py \
            --model xgboost \
            --xgb_lr 0.05 \
            --xgb_max_depth 10 \
            --xgb_n_estimators 300 \
            --save_models \
            2>&1 | tee logs/training_xgb_$(date +%Y%m%d_%H%M%S).log
        ;;
    3)
        echo ""
        echo "=============================================================================="
        echo "RUNNING: Advanced Training - LightGBM"
        echo "Command: python train_model_advanced.py --model lightgbm --lgb_lr 0.05 --lgb_max_depth 10 --lgb_n_estimators 300 --save_models"
        echo "=============================================================================="
        python train_model_advanced.py \
            --model lightgbm \
            --lgb_lr 0.05 \
            --lgb_max_depth 10 \
            --lgb_n_estimators 300 \
            --save_models \
            2>&1 | tee logs/training_lgb_$(date +%Y%m%d_%H%M%S).log
        ;;
    4)
        echo ""
        echo "=============================================================================="
        echo "RUNNING: Advanced Training - Both Models (Production)"
        echo "Command: python train_model_advanced.py --model both --xgb_lr 0.05 --lgb_lr 0.05 --save_models"
        echo "=============================================================================="
        python train_model_advanced.py \
            --model both \
            --xgb_n_estimators 300 \
            --xgb_max_depth 10 \
            --xgb_lr 0.05 \
            --lgb_n_estimators 300 \
            --lgb_max_depth 10 \
            --lgb_lr 0.05 \
            --val_split 0.8 \
            --random_seed 42 \
            --save_models \
            2>&1 | tee logs/training_production_$(date +%Y%m%d_%H%M%S).log
        ;;
    5)
        echo "Skipping training"
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "=============================================================================="
echo "COMPLETED"
echo "=============================================================================="
