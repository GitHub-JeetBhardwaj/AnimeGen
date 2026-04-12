# 🎌 AnimeGen: Reality to 2D Style Transfer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg?logo=opencv&logoColor=white)

**AnimeGen** is a full-stack generative AI and computer vision framework designed to transform real-world photographs into high-contrast, 2D cel-shaded anime art. 

This repository contains both the **Flask-based web application** (featuring a custom Neo-Manga UI) and the **PyTorch training pipeline** used to develop the generative models.

🌍 **Live Demo:** [thejeetbhardwaj-animegen.hf.space](https://thejeetbhardwaj-animegen.hf.space/)

---

## ✨ System Architecture & Pipelines

The application features three distinct processing modes, allowing users to choose between deterministic computer vision math and hallucinated generative AI.

### ⚙️ SYSTEM .01: Generative GAN (256px Base)
* **Architecture:** Text-Guided U-Net Generator with Inverted Residual Blocks and Cross-Attention layers.
* **Process:** Images are statically resized to 256x256, processed through the GAN using CLIP text embeddings, and upscaled using **Real-ESRGAN (Anime 6B)**. 
* **Output:** Hallucinates vibrant landscapes with pencil-like outlines.

### ⚙️ SYSTEM .02: OpenCV Posterization
* **Architecture:** Deterministic Computer Vision.
* **Process:** Applies bilateral filtering, boosts color saturation mathematically, applies color banding (posterization), and overlays an adaptive threshold edge-map.
* **Output:** Distinct, flat shading layers resembling classic 2D cel-shaded artwork without AI hallucination.

### 🔥 SYSTEM .03: Dynamic U-Net (SOTA)
* **Upcoming Architecture:** Advanced U-Net with custom layer-1 skip connections.


---

## 📊 Custom Anime Dataset

The generative models in this repository were trained on a highly curated, custom dataset created specifically for this project.

🔗 **Dataset Link:** [Anime Movies Dataset on Kaggle](https://www.kaggle.com/datasets/jeetbhardwaj01/anime-movies/data)

To ensure the models learned from the absolute best visual data, I built a custom Python/OpenCV extraction script that curated frames directly from visually stunning anime films (e.g., *Suzume*, *Your Name*). The dataset pipeline automatically filtered out motion blur (using Laplacian variance) and pitched-black transitions, resulting in thousands of perfectly center-cropped, high-quality reference images.

---

## 🧠 Model Training & Technical Highlights

The custom model was trained from scratch.

### 1. Hybrid 4-Channel Input
The generator does not just take RGB images. It takes a **4-channel tensor** consisting of:
* `Channel 1`: A normalized Canny edge stencil.
* `Channels 2-4`: A bilateral-filtered color hint.

This explicitly forces the generator to respect the structural boundaries of the original image before calculating loss.

### 2. Multi-Modal Loss Function
To prevent GAN "melting" and ensure crisp cel-shading, the training loop utilizes a custom weighted loss matrix:
* **L1 Pixel Anchoring (`x4.0`):** Forces the generator to respect exact geometry and lighting.
* **Deep Content Loss (`x3.0`):** Uses VGG19 (`relu1_2` and `relu2_2`) to anchor high-frequency sharp edges and mid-level structures.
* **Style Loss (`x2.0`):** Gram matrix matching against the target anime dataset.
* **Directional CLIP Loss (`x1.0`):** Cosine similarity between image direction vectors and text direction vectors to push the output toward specific studio aesthetics.

### 3. Final-Layer Skip Connection
A critical modification to standard U-Nets: The original 4-channel input is concatenated directly into the final decoder layer (`32 + 4 channels`), enforcing strict structural preservation while allowing the network to hallucinate styling.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/AnimeGen.git](https://github.com/yourusername/AnimeGen.git)
cd AnimeGen
```

### 2. Install Dependencies
Ensure you have an NVIDIA GPU (CUDA) for optimal inference speed.
```bash
pip install -r requirements.txt
```

### 3. Model Weights
Place your trained PyTorch `.pth` files in the root directory:
* `perfect.pth`


### 4. Run the Application
```bash
python app.py
```
The web app will launch at `http://127.0.0.1:5000`

---

## 👨‍💻 Author

**Jeet Bhardwaj**  
AI & Data Science Engineer  

- LinkedIn: https://linkedin.com/in/thejeetbhardwaj  
- GitHub: https://github.com/GitHub-JeetBhardwaj  
- HuggingFace: https://huggingface.co/TheJeetBhardwaj  
- Email: thejeetbhardwaj@gmail.com  
