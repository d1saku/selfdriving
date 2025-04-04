This is a Raspberry Pi Self-Driving Car

A TensorFlow-based autonomous vehicle platform running on Raspberry Pi.

## Overview

This project implements a small-scale self-driving car. 

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/self-driving-car.git
   cd self-driving-car
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure hardware connections in `config/hardware_config.json`

## Usage

### Training

To train the model:

```
python src/training/train.py --data_dir data/processed --config config/model_config.json
```

## Project Structure

The project follows a modular structure:

- `data/`: Raw and processed data, saved models
- `src/`: Source code
  - `preprocessing/`: Data preprocessing tools
  - `model/`: Neural network architecture
  - `training/`: Training scripts
- `utils/`: Utility functions
- `config/`: Configuration files
- `notebooks/`: Notebooks

## Model Architecture

The neural network uses a convolutional architecture inspired by NVIDIA's PilotNet
