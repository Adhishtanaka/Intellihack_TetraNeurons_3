# Qwen2.5-3b - Fine-Tuning Model

This repository contains the fine-tuned language model using **Unsloth** framework for fast and efficient training. The model is based on the **Qwen2.5-3B** architecture and has been fine-tuned on a custom dataset for various language generation tasks. The model supports **LoRA adapters** for efficient parameter updates.

## Model (unsloth.Q4_K_M.gguf)

- **Base Model**: Qwen2.5-3B-Instruct
- **Max Sequence Length**: 2048 tokens
- **Precision**: Uses 4-bit quantization for efficient memory usage.

if you cant downlaod this from here use my googledrive link : https://drive.google.com/file/d/1LbxKpNDZtCynpAR93coywIsSeI9M0G9H/view?usp=sharing

## Model fine-tuning Configuration (finetune_script & inference_script.ipynb)

- **LoRA Configuration**: 
  - **LoRA Rank**: 16 (Optimized rank for low-rank adaptation)
  - **Target Modules**: Includes key projection layers such as `q_proj`, `k_proj`, `v_proj`, and more for efficient fine-tuning.
  - **LoRA Alpha**: 16
  - **LoRA Dropout**: 0
  - **Bias**: No bias applied during training.
  - **Gradient Checkpointing**: Enabled for memory optimization during training.

- **Training Settings**:
  - **Max Steps**: Set to **150** (increased from 60 for better training due to the small dataset size).
  - **Evaluation Strategy**: Validation enabled during training with `eval_steps=25`, ensuring periodic evaluations every 25 steps.
 
Before running the finetune_script & inference_script.ipynb, ensure that the dataset (new.json) is also located in the same folder. The training process took around 40 minutes to complete and saved the file in GGuff format.

<img src="screenshots/train.png" alt="training" width="900">


## Dataset (New.json)

- The dataset used for fine-tuning was **synthetically generated** using the **Gemini API**, which helped create a diverse range of examples based on custom input prompts.
- The dataset is formatted with instructions and outputs, which are converted into the Alpaca-style format.
- Since the dataset is smaller, the number of training steps has been **increased** to ensure adequate model convergence without harming the functionality of the model.

## inference UI

This guide should help you set up and run the inference UI using **Chainlit**. 

<img src="screenshots/UI.png" alt="training" width="900">

### Step-by-Step Setup

#### 1. Create a Virtual Environment

```bash
# Navigate to your project directory
cd inference UI

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Requirements

```bash
# Install from requirements.txt
pip install -r requirements.txt

```

#### 3. Run Your Chainlit App

```bash
# Run the app
chainlit run index.py
```

This will start a local server, typically at http://localhost:8000, where you can interact with your inference UI.

## Rag-Script

This guide should help you set up and run the inference UI using **Chainlit** to communicate with gguf. 

### Step-by-Step Setup

#### 1. Create a Virtual Environment

```bash
# Navigate to your project directory
cd inference UI

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Requirements

```bash
# Install from requirements.txt
pip install -r requirements.txt

```

#### 3. Run Your Chainlit App

```bash
# Run the app
chainlit run index.py
```

This will start a local server, typically at http://localhost:8000, where you can interact with your inference UI.

## Setting Up a RAG System with Langchain, ChromaDB, and Chainlit

This guide walks through the process of setting up a Retrieval-Augmented Generation (RAG) system using Langchain and ChromaDB with a Chainlit UI.


### Step-by-Step Setup

#### 1. Create Project Structure

```bash

cd Rag-Script
```

#### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Create and Install Requirements

install the requirements:

```bash
pip install -r requirements.txt
```

#### 4. Add Documents to the Knowledge Base

Add your Markdown (.md) or text (.txt) files to the `documents/` folder. These will be used as the knowledge base for your RAG system.


#### 5. Run Your RAG Application

```bash
# Run the app
chainlit run index.py
```

#### 6. Using the RAG System

1. **Wait for the initialization message: "✅ RAG system initialized successfully!"**
   
   <img src="screenshots/wait.gif" alt="training" width="800">
   
2. **Once initialized, you can ask questions about the content of your documents**
   
   <img src="screenshots/Rag.png" alt="training" width="800">
   
3. **The system will retrieve relevant information from your documents and generate responses**




