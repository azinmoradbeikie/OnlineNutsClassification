# OnlineNutsClassification


we propose an online classification system for walnut samples from the Lara and Chandler cultivars, void and caliber, leveraging spike-triggered acoustic sensing and spectro-temporal sample description. The proposed solutions are based on a low-cost mechanical structure in which individual walnuts slide through a guided channel and impact a metallic plate, generating distinct acoustic signatures. A spike detection strategy is applied to the continuous audio stream to isolate high-energy impact events, allowing for consistent and non-invasive acquisition of time-aligned segments. From these segments, a comprehensive set of spectral and perceptual features is extracted, including Fast Fourier Transform (FFT) energy bands and Mel-Frequency Cepstral Coefficients (MFCCs). These features are then used to train a Random Forest classifier offline, which is later deployed for real-time prediction of nut quality classes. 
## Features
- Feature extraction from audio signals
- R
## Installation
```bash
git clone https://github.com/azinmoradbeikie/OnlineNutsClassification.git
cd OnlineNutsClassification
pip install -r requirements.txt

## Usage
```bash
python main.py
