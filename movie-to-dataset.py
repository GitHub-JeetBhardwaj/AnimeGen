import cv2
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "Planet-Earth-III-(2024)-Season-3-Hindi-Dubbed-Complete-Series--720p-[Orgmovies].mp4"      # Path to your 1080p/4K video file
OUTPUT_DIR = "planet_earth_3"   # Where the good frames will be saved
TARGET_SIZE = 256                    # Set to 256 or 512 based on your GPU
FRAME_INTERVAL_SEC = 2               # Extract 1 frame every 2 seconds

# Trimming Config
SKIP_MINUTES_START = 5               # Skip the first 5 minutes (opening/intro)
SKIP_MINUTES_END = 5                 # Skip the last 5 minutes (credits)

# Filtering Thresholds
BLUR_THRESHOLD = 100.0               # Higher = stricter blur filter (rejects more)
BRIGHTNESS_THRESHOLD = 40.0          # Reject frames where average pixel value is too dark

def setup_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def is_blurry(image_gray, threshold):
    """Calculates the variance of the Laplacian to detect blur."""
    variance = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return variance < threshold

def is_too_dark(image_gray, threshold):
    """Checks the average brightness of the image."""
    mean_brightness = np.mean(image_gray)
    return mean_brightness < threshold

def resize_and_crop(image, target_size):
    """Resizes the shortest edge to target_size, then center crops."""
    h, w = image.shape[:2]
    
    # 1. Resize shortest edge
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Center Crop
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
    
    return cropped

def extract_dataset():
    setup_directory(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * FRAME_INTERVAL_SEC)
    
    # Calculate start and end frames
    start_frame = int(SKIP_MINUTES_START * 60 * fps)
    end_frame = int(total_frames - (SKIP_MINUTES_END * 60 * fps))
    
    if start_frame >= end_frame:
        print("Error: The video is too short to skip that many minutes.")
        return

    print(f"Video FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Skipping first {SKIP_MINUTES_START} mins (starting at frame {start_frame})")
    print(f"Skipping last {SKIP_MINUTES_END} mins (ending at frame {end_frame})")
    
    # Jump directly to the starting frame to save time
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved_count = 0
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break # End of video or read error
            
        # Only process frames at the specified interval
        if current_frame % frame_interval == 0:
            # 1. Resize and Crop to perfect square
            processed_frame = resize_and_crop(frame, TARGET_SIZE)
            
            # Convert to grayscale for metric calculations
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Filter out dark/pitch-black scenes
            if is_too_dark(gray, BRIGHTNESS_THRESHOLD):
                current_frame += 1
                continue
                
            # 3. Filter out motion blur
            if is_blurry(gray, BLUR_THRESHOLD):
                current_frame += 1
                continue
                
            # 4. Save the frame if it passes all tests
            output_path = os.path.join(OUTPUT_DIR, f"suzume_frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_path, processed_frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} high-quality frames so far...")

        current_frame += 1

    cap.release()
    print(f"Extraction complete! Successfully saved {saved_count} clean frames to {OUTPUT_DIR}.")

extract_dataset()