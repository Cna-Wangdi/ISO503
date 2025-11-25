**ISY503 – Intelligent Systems**
**NLP Sentiment Analysis Project (Bidirectional LSTM + Streamlit App)**

This repository contains the full implementation for our ISY503 Assessment 3 project, where we developed a Natural Language Processing      (NLP) Sentiment Analysis System using a Bidirectional LSTM model trained on the Amazon Review Dataset. The project includes the complete training pipeline, a saved deep learning model, and a Streamlit-based web application for real-time sentiment prediction.

**Project Structure:**/

    ISY503_NLP   
      ├── sentiment_pipeline.py      
      ├── app.py      
      ├── sentiment_lstm_model.keras      
      ├── tokenizer.pickle      
      ├── domain_sentiment_data/      
      │ └── sorted_data_acl/      
      │ ├── books/      
      │ ├── dvd/      
      │ ├── electronics/      
      │ └── kitchen_&_housewares/      
      └── README.md             

**1. Project Overview**

This project implements a binary sentiment classifier that predicts whether a customer review is Positive or Negative.

We used:

    * Multi-Domain Sentiment Datase (Books / DVD / Electronics / Kitchen domains): https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
    
    * Text preprocessing (cleaning, tokenisation, padding)
    
    * Bidirectional LSTM deep learning model
    
    * Class balancing
    
    * Train/Validation/Test split
    
    * Streamlit web UI for demonstration

The system is lightweight, easy to use, and demonstrates key concepts of Intelligent Systems and modern NLP.

**2. Model Architecture**

The final model uses a Bidirectional LSTM with:

    * Embedding layer
    
    * BiLSTM (256 units)
    
    * GlobalMaxPool1D
    
    * Dense (128 → 64) with ReLU
    
    * Dropout regularisation
    
    * Sigmoid output for binary classification

Features:
    * Handles long-range dependencies
    * Learns both forward and backward context
    * Strong performance on sentiment classification

**3. Installation & Setup**

Step 1 — Clone the Repository

    git clone https://github.com/USERNAME/ISY503_NLP.git
    
    cd ISY503_NLP

Step 2 — Install Dependencies

    pip install -r requirements.txt

Step 3 — Prepare the Dataset

Download or extract the dataset:/

Step 4 — Train the Model
    python sentiment_pipeline.py

This will generate:
    - sentiment_lstm_model.keras
    - tokenizer.pickle

**4. Running the Web Application**

    Start the Streamlit app:
        streamlit run app.py
    
    Your browser will open:
        http://localhost:8501
    
**5. Dataset Information**

We used the Amazon Review Polarity Dataset, containing labelled positive and negative reviews across four domains:

    * Books
    
    * DVD
    
    * Electronics
    
    * Kitchen & Housewares
    
A randomised, cleaned version was used to avoid bias and noise.

Source:
Dredze, M. (n.d.). Sentiment datasets. Johns Hopkins University.
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/

**6. Ethical Considerations**

Our group considered the following ethical issues:

    * Bias: model may favour positive reviews due to imbalance
    
    * Misclassification: harmful in real-world scenarios
    
    * Data limitations: only English customer reviews
    
    * Transparency: cleaning steps, limitations, and confidence scores displayed
    
    * Accountability: model should not be used for high-risk decisions

**7. Final Notes**

    This project demonstrates a complete Intelligent Systems pipeline using modern NLP techniques and provides a fully functional sentiment analysis interface. It is suitable for academic demonstration, future extension, and practical learning.
