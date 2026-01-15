# Handwritten Digit Recognition Neural Network

This project implements a **fully connected neural network from scratch using NumPy** to classify handwritten digits (MNIST-style). It also includes a **FastAPI backend** to expose the trained model for inference, allowing integration with a web-based frontend (e.g., a React canvas).  

---

## Features

- **Neural network from scratch**:  
  - Input: 28×28 grayscale images (flattened to 784)  
  - Hidden layer: 10 neurons, ReLU activation  
  - Output: 10 neurons, softmax  
- **Training pipeline**:
  - Custom gradient descent
  - One-hot encoding of labels
- **Model persistence**: Saves weights as `.npy` files for easy inference
- **FastAPI backend**: Exposes a `/predict` endpoint
- **Confidence output**: Returns the predicted digit and its probability

---

## Folder Structure

```text
Handwritten-Digit-NN/          
├── README.md                  
├── train.py                    
├── model.py                    
├── main.py                     
├── data/                       
│   └── train.csv
├── weights/                    
│   ├── W1.npy
│   ├── b1.npy
│   ├── W2.npy
│   └── b2.npy
└── frontend/                   
    └── ...

```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/szymonwyro/Neural-Network.git
cd Neural-Network
```

### 2. Install dependencies

```bash
pip install numpy fastapi uvicorn pandas matplotlib
python -m uvicorn main:app --reload
```

### 3. Run FastAPI backend

```bash
python -m uvicorn main:app --reload
```


