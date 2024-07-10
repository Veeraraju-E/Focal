from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from PIL import Image, UnidentifiedImageError
import piexif
import json
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import hashlib
import os
import urllib.request

# Global variables
TAGS_FILE = os.path.join(settings.BASE_DIR, 'tags.xlsx')
JSON_FILE = os.path.join(settings.BASE_DIR, 'image_tags.json')
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'uploads')
EXTERNAL_TAGS_FILE = os.path.join(settings.BASE_DIR, 'tags.txt')
images = []
current_image_index = 0
directory = ''

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load or create the Excel file
if os.path.exists(TAGS_FILE):
    df = pd.read_excel(TAGS_FILE)
else:
    df = pd.DataFrame(columns=['Image Path', 'Tags'])
    df.to_excel(TAGS_FILE, index=False)

# Load or create the JSON file
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, 'r') as f:
        image_tags = json.load(f)
else:
    image_tags = {}

# Load labels for classification
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.load(urllib.request.urlopen(LABELS_URL))

# Load pre-trained model for feature extraction
model = resnet50(pretrained=True)
model.eval()

# Load external tags
def load_external_tags():
    if os.path.exists(EXTERNAL_TAGS_FILE):
        with open(EXTERNAL_TAGS_FILE, 'r') as f:
            tags = f.read().splitlines()
        return tags
    return []

external_tags = load_external_tags()

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_images(dir_path):
    global images
    images = []
    print(f"Loading images from directory: {dir_path}")
    for root, _, files in os.walk(dir_path):
        for file in files:
            if allowed_file(file):
                relative_path = os.path.relpath(os.path.join(root, file), start=dir_path)
                images.append(relative_path.replace('\\', '/'))
    print(f"Loaded images: {images}")

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()

def generate_tags_for_image(image_path):
    feature_vector = extract_features(image_path)
    predicted_indices = torch.topk(torch.tensor(feature_vector), 5).indices.tolist()
    predicted_labels = [labels[idx] for idx in predicted_indices]
    return predicted_labels

def save_metadata(image_path, tags):
    print(f"Attempting to save metadata for image: {image_path}")
    try:
        image = Image.open(image_path)
        exif_dict = piexif.load(image.info.get('exif', b''))
        user_comment = tags.encode('utf-8')
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment, encoding="unicode")
        exif_bytes = piexif.dump(exif_dict)
        image.save(image_path, "jpeg", exif=exif_bytes)
        print(f"Successfully saved metadata for image: {image_path}")
    except UnidentifiedImageError as e:
        print(f"Error: Cannot identify image file {image_path}. {e}")
    except KeyError as e:
        print(f"Error: Missing EXIF data in {image_path}. {e}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def save_text_file(image_path, tags):
    try:
        # Generate the text file path with the same base name as the image
        text_file_path = os.path.splitext(image_path)[0] + '.txt'
        # Write the tags to the text file
        with open(text_file_path, 'w') as f:
            f.write(tags)
        print(f"Successfully saved text file for image: {text_file_path}")
    except Exception as e:
        print(f"Error saving text file for {image_path}: {e}")

def save_tags_to_json(image_path, tags):
    global image_tags
    feature_vector = extract_features(image_path)
    feature_key = hashlib.md5(feature_vector.tobytes()).hexdigest()  # Shortened hash key
    image_tags[feature_key] = tags.split(',')
    with open(JSON_FILE, 'w') as f:
        json.dump(image_tags, f)
    print(f"Successfully saved tags to JSON for image: {image_path}")

def save_tags_to_excel(image_path, tags):
    global df
    # Check if image_path already exists in DataFrame
    if image_path in df['Image Path'].values:
        df.loc[df['Image Path'] == image_path, 'Tags'] = tags
    else:
        new_row = {'Image Path': image_path, 'Tags': tags}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(TAGS_FILE, index=False)
    print(f"Successfully saved tags to Excel for image: {image_path}")

# Django views
# +---------------------------- The views ----------------------------+ 
def index(request):
    global directory, current_image_index, images
    if request.method == 'POST':
        directory = request.POST.get('directory')
        if os.path.isdir(directory):
            load_images(directory)
            current_image_index = 0
    ai_tags = generate_tags_for_image(os.path.join(directory, images[current_image_index])) if images else []
    print("in ''")
    print(current_image_index)
    return render(request, 'index.html', {
        'image': images[current_image_index] if images else None,
        'directory': directory,
        'ai_tags': ai_tags,
        'external_tags': external_tags,
    })

def uploaded_file(request, filename):
    global directory
    return HttpResponse(directory + '/' + filename)

def tag_image(request):
    if request.method == 'POST':
        tags = request.POST.get('tags')
        image_path = request.POST.get('image_path')
        store_option = request.POST.get('store_option')
        absolute_image_path = os.path.join(directory, image_path)
        print(f"Received tags: {tags}, store option: {store_option}, for image: {absolute_image_path}")
        if store_option == 'metadata':
            save_metadata(absolute_image_path, tags)
            save_tags_to_excel(absolute_image_path, tags)  # Save tags to Excel when image metadata option is selected
        elif store_option == 'text_file':
            save_text_file(absolute_image_path, tags)
        save_tags_to_json(absolute_image_path, tags)
    return redirect('index')

def next_image(request):
    global current_image_index
    if len(images) > 0:
        current_image_index = (current_image_index + 1) % len(images)
    return redirect('index')

def prev_image(request):
    global current_image_index
    if len(images) > 0:
        current_image_index = (current_image_index - 1) % len(images)
    return redirect('index')

def find_tags(request):
    assigned_tags = []
    unassigned_tags = []
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, 'find_tags.html', {'assigned_tags': assigned_tags, 'unassigned_tags': unassigned_tags})
        file = request.FILES['file']
        if file:
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            uploaded_file_url = fs.url(filename)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            assigned_tags, unassigned_tags = get_tags_for_image(file_path)
            return render(request, 'find_tags.html', {
                'filename': filename,
                'assigned_tags': assigned_tags,
                'unassigned_tags': unassigned_tags,
            })
    return render(request, 'find_tags.html', {'assigned_tags': assigned_tags, 'unassigned_tags': unassigned_tags})

def update_tags(request):
    if request.method == 'POST':
        tags = request.POST.get('tags')
        filename = request.POST.get('filename')
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        save_tags_to_json(file_path, tags)
        save_tags_to_excel(file_path, tags)  # Save updated tags to Excel
        return redirect('find_tags')
    return redirect('find_tags')

# Helper function

def get_tags_for_image(image_path):
    feature_vector = extract_features(image_path)
    feature_key = hashlib.md5(feature_vector.tobytes()).hexdigest()  # Shortened hash key
    assigned_tags = image_tags.get(feature_key, [])
    predicted_tags = generate_tags_for_image(image_path)
    unassigned_tags = [tag for tag in external_tags if tag not in assigned_tags]
    return assigned_tags, unassigned_tags
