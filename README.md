# Emotion Classification Models 🎭

Fine-tuned transformer models for classifying text into 4 emotion categories: **Joy**, **Sadness**, **Neutral**, and **Anger**.

## 📊 Models Overview

This repository contains two fine-tuned emotion classification models:

### 1. **j-hartmann Model** 
- **Base Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Test Accuracy**: 78.21%
- **F1 Score (Macro)**: 77.96%
- **Best Performance**: Anger detection (F1: 87.28%)

### 2. **CardiffNLP Model**
- **Base Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Test Accuracy**: 79.04%
- **F1 Score (Macro)**: 78.73%
- **Best Performance**: Anger detection (F1: 88.79%)
- **Strength**: Optimized for Twitter/social media text

## 🎯 Emotion Categories

| Label | Emotion | Description |
|-------|---------|-------------|
| 0 | Joy | Positive emotions, happiness, excitement |
| 1 | Sadness | Negative emotions, disappointment, grief |
| 2 | Neutral | Factual statements, neutral tone |
| 3 | Anger | Frustration, anger, irritation |

## 📈 Performance Comparison

| Model | Accuracy | F1 (Macro) | Joy F1 | Sadness F1 | Neutral F1 | Anger F1 |
|-------|----------|------------|--------|------------|------------|----------|
| j-hartmann | 78.21% | 77.96% | 72.30% | 75.64% | 76.62% | 87.28% |
| CardiffNLP | 79.04% | 78.73% | 71.33% | 77.42% | 77.38% | 88.79% |

**Recommendation**: Use **CardiffNLP model** for Twitter/social media text, **j-hartmann model** for general text.

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch numpy pandas
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model (choose one)
MODEL_PATH = "./emotion_classifier_jhartmann/final_model"  # or cardiffnlp
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Emotion labels
EMOTIONS = {0: "Joy", 1: "Sadness", 2: "Neutral", 3: "Anger"}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)[0]
    pred_label = torch.argmax(probs).item()
    confidence = probs[pred_label].item()
    
    return EMOTIONS[pred_label], confidence

# Example
text = "I'm so happy today! Everything is going great!"
emotion, confidence = predict_emotion(text)
print(f"Emotion: {emotion} (Confidence: {confidence:.2%})")
```

## 📁 Repository Structure

```
emotion-classification/
│
├── README.md                          # This file
├── LICENSE                            # License file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   ├── sentiment_classifier_cardiffnlp.ipynb
│   └── sentiment_classifier_jhartmann.ipynb
│
├── models/
│   ├── emotion_classifier_jhartmann/
│   │   └── final_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── special_tokens_map.json
│   │       ├── vocab.json
│   │       ├── merges.txt
│   │       └── model_config.json
│   │
│   └── sentiment_classifier_cardiffnlp/
│       └── final_model/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── special_tokens_map.json
│           ├── vocab.json
│           ├── merges.txt
│           └── model_config.json
│
├── scripts/
│   ├── train_jhartmann.py            # Training script for j-hartmann
│   ├── train_cardiffnlp.py           # Training script for CardiffNLP
│   └── predict.py                    # Inference script
│
└── data/
    └── emotions-dataset.csv           # Training dataset (if permitted)
```

## 🔧 Training Details

### Dataset
- **Size**: 35,066 samples
- **Distribution**: 
  - Joy: 26.27%
  - Sadness: 27.18%
  - Neutral: 18.29%
  - Anger: 28.27%
- **Split**: 80% train, 10% validation, 10% test (stratified)

### Training Configuration
- **Epochs**: 4-5
- **Batch Size**: 32 (train), 64 (eval)
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with weight decay (0.01)
- **Loss Function**: Cross-entropy with class weights
- **Hardware**: NVIDIA Tesla T4 GPU

### Key Techniques
- ✅ Transfer learning from pre-trained models
- ✅ Class weight balancing for imbalanced data
- ✅ Early stopping (patience: 2)
- ✅ Stratified train/validation/test splits
- ✅ Dynamic padding with data collator

## 📊 Detailed Results

### Confusion Matrix (CardiffNLP Model)

```
              Joy    Sadness  Neutral  Anger
Joy           612      152      121     36
Sadness        88      785       45     35
Neutral        61       52      520      8
Anger          34       86       17    855
```

### Common Misclassifications
- **Joy ↔ Sadness**: Most common error (20.7%)
- **Joy ↔ Neutral**: Second most common (16.5%)
- **Reason**: Subtle emotional nuances, sarcasm, mixed sentiments

## 🎓 Use Cases

- 📱 Social media sentiment monitoring
- 💬 Customer feedback analysis
- 🎮 Gaming chat moderation
- 📧 Email categorization
- 🤖 Chatbot emotion detection
- 📊 Market research and opinion mining

## ⚠️ Limitations

- Models trained on Twitter-style text (short, informal)
- May struggle with:
  - Sarcasm and irony
  - Mixed emotions in longer texts
  - Domain-specific jargon
  - Non-English text
- Anger/Joy confusion in some contexts

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Base Models**:
  - [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
  - [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Framework**: Hugging Face Transformers
- **Dataset**: Emotions Dataset

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this useful, please star the repository!**
