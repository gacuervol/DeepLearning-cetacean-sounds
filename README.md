# 🎤 Cetacean Vocalization Recognition | Deep Learning & Transfer Learning  
*Classification of marine mammal sounds using YAMNet embeddings and data augmentation techniques*  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow) ![YAMNet](https://img.shields.io/badge/Transfer_Learning-YAMNet-yellow) ![Librosa](https://img.shields.io/badge/Audio_Librosa-0.10-brightgreen) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-red)

## 📌 **Project Overview**  
**Objective**: Develop a robust classifier for cetacean vocalizations by:  
- **Preprocessing**: Spectral noise reduction and silence removal  
- **Feature Extraction**: Fixed YAMNet embeddings (1024-dim vectors)  
- **Data Augmentation**: Time/Frequency masking to address class imbalance  
- **Classification**: Custom dense neural network (512 → 55 units)  

**Key Achievements**:  
✅ **98% test accuracy** across 55 marine species  
✅ Processed **15,563 audio samples** with imbalanced classes  
✅ Implemented end-to-end pipeline from raw audio to predictions  


## 🛠️ Technical Stack  
```python
# Core Architecture
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')  # Transfer Learning
classifier = Sequential([
    Dense(512, activation='relu', input_shape=(1024,)),
    Dense(55)  # Number of species
])

# Audio Processing
audio = nr.reduce_noise(y=audio, sr=16000, stationary=True)  # Noise reduction
frames = librosa.util.frame(audio, frame_length=800, hop_length=400)  # Segmentation
```
## � **Technical Methodology**  
### 🔧 Transfer Learning (Feature Extraction Approach)  
```python
# Frozen YAMNet as feature extractor (no fine-tuning)
yamnet = hub.load('google/yamnet/1')  # Weights locked
embeddings = yamnet(wav_data)[1]     # 1024-dim embeddings

# Trainable classifier
model = Sequential([
    Dense(512, activation='relu', input_shape=(1024,)),
    Dense(55, activation='softmax')  # Multi-class output
])
```

**Why This Approach?**  
✔ **Computational Efficiency**: Avoids training CNN from scratch  
✔ **Generalization**: Leverages YAMNet's pre-trained acoustic patterns  
✔ **Data Efficiency**: Embeddings reduce need for massive datasets  


## 📊 **Performance Metrics**  
### 1. Classification Report (Test Set)  
| Metric          | Value  |  
|-----------------|--------|  
| Accuracy        | 98.0%  |  
| Macro F1-Score  | 97.4%  |  
| Avg Recall      | 97.2%  |  

### 2. Processing Pipeline
![pipeline](https://github.com/gacuervol/DeepLearning-cetacean-sounds/blob/main/figures/pipeline.png)
*Illustrating the distinct stages of the cetacean vocalization recognition pipeline.*

### 3. t-SNE Visualization  
![t-SNE Plot](https://github.com/gacuervol/DeepLearning-cetacean-sounds/blob/main/figures/tsne.png)
*Clear clustering of acoustically similar species*  

### 4. Training Dynamics  
![Loss Curves](https://github.com/gacuervol/DeepLearning-cetacean-sounds/blob/main/figures/training_plot.png)
*Early stopping at epoch 130 (val_loss=0.1243)*  


## 📂 Repository Structure  
```text
/Data
├── df_audios.csv              
/Notebooks
├── 1_business_data_load.ipynb
├── 2_exploratory_data_analysis.ipynb
├── 3_experimental_set_up.ipynb
├── 4_modeling.ipynb
├── Filtered_signal.ipynb
├── Harmonic-Percusiv.ipynb
├── Proyecto_Final.ipynb
/figures
├── training_plot.png           
├── tsne.png 
```

## 🚀 How to Use  
### 1. Install dependencies:  
```bash
pip install -r requirements.txt  # Includes TensorFlow, Librosa, Noisereduce
```

### 2. Run inference on new audio:  
```python
from inference import predict_species
probabilities = predict_species("dolphin.wav")  # Returns class probabilities
```

## 🧠 Key Technical Challenges  
- **Audio Variability**: Solved with spectral noise reduction (`noisereduce` library)  
- **Class Imbalance**: Addressed via synthetic data augmentation (time/frequency masking)  
- **Embedding Optimization**: Fine-tuned YAMNet's 1024-dim output with dense layers  

## 📜 Research Applications  
- Marine Conservation: Endangered species monitoring  
- Bioacoustics: Migration pattern analysis  
- Oceanography: Anthropogenic noise impact studies
  

## 🔗 Connect  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/giovanny-alejandro-cuervo-londo%C3%B1o-b446ab23b/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/Giovanny-Cuervo-Londono)  
[![Email](https://img.shields.io/badge/Email-giovanny.cuervo101%40alu.ulpgc.es-D14836?style=for-the-badge&logo=gmail)](mailto:giovanny.cuervo101@alu.ulpgc.es)  

> 🌴 **Research Opportunities**:  
> - Open to collaborations  
> - Contact via LinkedIn for consulting  
