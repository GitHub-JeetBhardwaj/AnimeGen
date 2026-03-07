import os
import uuid
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import numpy as np
from skimage import exposure
import cv2
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Fix for torchvision
import sys
try:
    import torchvision.transforms.functional_tensor as F_t
except ImportError:
    import torchvision.transforms.functional as F_t
    sys.modules['torchvision.transforms.functional_tensor'] = F_t

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = Flask(__name__)
app.secret_key = "super_secret_anime_key"

# ==========================================
# ABSOLUTE PATHS FOR STATIC FOLDERS
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# CONFIG & GAN SETUP (Mode 1 - UPDATED)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

# NEW: Updated Model Weights & Prompt
CHECKPOINT_PATH = "perfect.pth" 
PROMPT = "2D vector art, flat cel shading, heavy outline strokes, hand-drawn anime background art Vibrant Makoto Shinkai painting, zero photorealism, anime illustration Studio Ghibli style cinematic anime scene, painted background Flat 2D anime art with glowing lighting, pure illustration"
OUTLINEWIDTH = 1
REFERENCE_ANIME_IMG = None 

class CrossAttention(nn.Module):
    def __init__(self, in_dim, text_dim=512):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key = nn.Linear(text_dim, in_dim//8)
        self.value = nn.Linear(text_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, text_embeds):
        b, c, w, h = x.size()
        q = self.query(x).view(b, -1, w*h).permute(0, 2, 1)
        k = self.key(text_embeds).unsqueeze(2)
        v = self.value(text_embeds).unsqueeze(2)
        attn = torch.softmax(torch.bmm(q, k), dim=1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, w, h)
        return x + self.gamma * out

class ConvNormLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.m(x)

class InvertedResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.m = nn.Sequential(
            ConvNormLReLU(channels, channels*2, 1, 1, 0),
            nn.Conv2d(channels*2, channels*2, 3, 1, 1, groups=channels*2, bias=False),
            nn.InstanceNorm2d(channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels*2, channels, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x): return x + self.m(x)

class TextGuidedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvNormLReLU(4, 32, 7, 1, 3)
        self.enc2 = ConvNormLReLU(32, 64, 3, 2, 1)
        self.enc3 = ConvNormLReLU(64, 128, 3, 2, 1)
        self.res_blocks = nn.Sequential(*[InvertedResBlock(128) for _ in range(4)])
        self.cross_attn = CrossAttention(128, text_dim=512)
        self.up1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1_conv = ConvNormLReLU(128 + 64, 64, 3, 1, 1)
        self.up2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2_conv = ConvNormLReLU(64 + 32, 32, 3, 1, 1)
        
        # NEW: Input channels changed to 32 + 4 to accommodate the skip connection
        self.final = nn.Sequential(
            nn.Conv2d(32 + 4, 3, 7, 1, 3), 
            nn.Tanh()
        )

    def forward(self, x, text_embeds):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.res_blocks(e3)
        b = self.cross_attn(b, text_embeds)
        u1 = self.up1_up(b)
        u1 = self.up1_conv(torch.cat([u1, e2], dim=1))
        u2 = self.up2_up(u1)
        u2 = self.up2_conv(torch.cat([u2, e1], dim=1))
        
        # NEW: Concatenate input tensor `x` with `u2`
        out = self.final(torch.cat([u2, x], dim=1)) 
        return out

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def create_4ch_input(pil_img):
    img_resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img_np = np.array(img_resized)
    simplified = cv2.pyrMeanShiftFiltering(img_np, sp=10, sr=30)
    gray = cv2.cvtColor(simplified, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=70, threshold2=150)
    kernel = np.ones((2, 2), np.uint8) 
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = (edges / 255.0).astype(np.float32)
    edges_tensor = torch.from_numpy(edges).unsqueeze(0)
    edges_tensor = (edges_tensor - 0.5) / 0.5 
    hint = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    hint_tensor = base_transform(Image.fromarray(hint))
    input_4ch = torch.cat([edges_tensor, hint_tensor], dim=0).unsqueeze(0).to(DEVICE)
    return input_4ch

# Try to load models globally
try:
    print("Loading CLIP...")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    print(f"Loading Generator from {CHECKPOINT_PATH}...")
    G = TextGuidedGenerator().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'G_state_dict' in checkpoint:
        G.load_state_dict(checkpoint['G_state_dict'])
    else:
        G.load_state_dict(checkpoint)
    G.eval() 
    
    print("Loading Upscaler...")
    upscaler_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        model=upscaler_model,
        device=DEVICE,
        half=True if DEVICE == "cuda" else False 
    )
    
    text_inputs = clip_tokenizer([PROMPT], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_embeds = clip_text_model(**text_inputs).text_embeds
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Mode 1 models failed to load. Error: {e}")
    MODELS_LOADED = False

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def process_mode_1_gan(input_path, output_path):
    if not MODELS_LOADED:
        return False, "Models not loaded. Check server console."
    try:
        with torch.no_grad(): 
            img = Image.open(input_path).convert("RGB")
            
            # Histogram matching logic included (triggers if REFERENCE_ANIME_IMG is set)
            if REFERENCE_ANIME_IMG and os.path.exists(REFERENCE_ANIME_IMG):
                ref_img = Image.open(REFERENCE_ANIME_IMG).convert("RGB")
                img_np = np.array(img)
                ref_np = np.array(ref_img)
                matched_np = exposure.match_histograms(img_np, ref_np, channel_axis=-1)
                img = Image.fromarray(matched_np.astype('uint8'))
                
            orig_w, orig_h = img.size
            input_tensor_4ch = create_4ch_input(img)
            
            fake_anime = G(input_tensor_4ch, text_embeds)
            
            out_tensor = (fake_anime.squeeze(0) * 0.5) + 0.5 
            out_tensor = torch.clamp(out_tensor, 0, 1)
            out_np = out_tensor.permute(1, 2, 0).cpu().numpy() * 255.0 
            out_np = out_np.astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            
            upscaled_bgr, _ = upsampler.enhance(out_bgr, outscale=4)
            upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
            output_pil = Image.fromarray(upscaled_rgb)
            output_pil = output_pil.resize((orig_w, orig_h), Image.LANCZOS)
            
            color_enhancer = ImageEnhance.Color(output_pil)
            output_pil = color_enhancer.enhance(1.4) 
            contrast_enhancer = ImageEnhance.Contrast(output_pil)
            output_pil = contrast_enhancer.enhance(1.15)
            
            # NEW: Outline overlay disabled to match your script
            # output_pil = apply_anime_outline(img, output_pil, outline_width=OUTLINEWIDTH)
            
            output_pil.save(output_path)
            
            if os.path.exists(output_path):
                return True, "Success"
            else:
                return False, "Failed to save image to disk."
    except Exception as e:
        return False, str(e)

def process_mode_2_opencv(input_path, output_path, color_levels=6):
    try:
        img = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        smoothed = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.5  
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        step = 256 // color_levels
        posterized = np.uint8((vibrant // step) * step)
        
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        final_img = cv2.bitwise_and(posterized, edges_color)
        
        cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        
        if os.path.exists(output_path):
            return True, "Success"
        else:
            return False, "Failed to write image to disk using cv2."
    except Exception as e:
        return False, str(e)

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models')
def models_page():
    return render_template('models.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload/<int:mode>', methods=['GET', 'POST'])
def upload(mode):
    if mode == 3:
        flash("Mode 3 is Coming Soon!")
        return redirect(url_for('models_page'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            in_filename = f"in_{unique_id}_{filename}"
            out_filename = f"out_{unique_id}_{filename}"
            
            input_path = os.path.join(UPLOAD_FOLDER, in_filename)
            output_path = os.path.join(OUTPUT_FOLDER, out_filename)
            file.save(input_path)
            
            if mode == 1:
                success, msg = process_mode_1_gan(input_path, output_path)
            elif mode == 2:
                success, msg = process_mode_2_opencv(input_path, output_path)
                
            if success:
                return render_template('result.html', original=in_filename, result=out_filename)
            else:
                flash(f"Error processing image: {msg}")
                return redirect(request.url)
                
    return render_template('upload.html', mode=mode)

if __name__ == '__main__':
    app.run(debug=True, port=5000)