from django.shortcuts import render, redirect
from django.conf import settings
from django.http import FileResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
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
# UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'uploads')
images = []
current_image_index = 0
directory = ''
external_tags = {}

# Ensure upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# Load or create the Excel file
def load_tags_from_excel():
    if os.path.exists(TAGS_FILE):
        df = pd.read_excel(TAGS_FILE, engine='openpyxl')
    else:
        df = pd.DataFrame(columns=['Image Path', 'Tags', 'Species', 'Reference'])
        df.to_excel(TAGS_FILE, index=False, engine='openpyxl')

    for row in df.iloc[:, 1]:
        for tag in str(row).split(','):
            external_tags[tag] = external_tags.get(tag, 0) + 1
    return df

df = load_tags_from_excel()

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


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_images(dir_path):
    global images
    images = []
    print(f"Loading images from directory: {dir_path}")
    for root, _, files in os.walk(dir_path):
        for i, file in enumerate(files):
            if allowed_file(file):
                relative_path = os.path.relpath(os.path.join(root, file), start=dir_path)
                # print(i, relative_path.replace('\\', '/'))
                images.append(relative_path)
    for i, image in enumerate(images):
        print(i, image)
    return images

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transformations(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()

def generate_tags_for_image(image_path):
    feature_vector = extract_features(image_path)
    predicted_indices = torch.topk(torch.tensor(feature_vector), 5).indices.tolist()
    # print(predicted_indices)
    predicted_labels = [labels[idx] for idx in predicted_indices]
    return predicted_labels

def save_metadata(image_path, tags, species, referenec):
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


def save_info_to_text_file(image_path, tags, species, reference):

    new_tags = [tag.strip() for tag in tags.split(',')]

    for tag in new_tags:
        external_tags[tag] = external_tags.get(tag, 0) + 1


def save_tags_to_json(image_path, tags):
    global image_tags
    feature_vector = extract_features(os.path.join(directory, image_path))
    feature_key = hashlib.md5(feature_vector.tobytes()).hexdigest()  # Shortened hash key
    image_tags[feature_key] = tags.split(',')
    print(f"In save_tags_to_json, directory : {directory}, image_path : {image_path}")
    with open(JSON_FILE, 'w') as f:
        json.dump(image_tags, f)
    print(f"Successfully saved tags to JSON for image: {image_path}")

def save_info_to_excel(image_path, tags, species, reference):
    global df
    # Check if image_path already exists in DataFrame
    if image_path in df['Image Path'].values:
        df.loc[df['Image Path'] == image_path, 'Tags'] = tags
        df.loc[df['Image Path'] == image_path, 'Species'] = species
        df.loc[df['Image Path'] == image_path, 'Reference'] = reference
    else:
        new_row = {'Image Path': image_path, 'Tags': tags}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(TAGS_FILE, index=False)
    print(f"Successfully saved info to Excel for image: {image_path}")

def get_existing_info(image_path):
    try:
        df = load_tags_from_excel()
        for i, row in enumerate(df.iloc[:, 0]):
            if image_path == row:
                return df.iloc[i, 1:4]  # info tags for the image
    except Exception as e:
        print(f"Error retrieving tags for {image_path}: {e}")
    return ["", "", ""]

# +---------------------------- The views ----------------------------+ 
def home(request):
    print("in /")
    global directory, current_image_index, images, external_tags
    if request.method == 'POST':
        directory = request.POST.get('directory')
        if os.path.isdir(directory):
            images = load_images(directory)
            if images:
                current_image_index = 0
                return redirect("index")
            else:
                messages.error("No images found in this directory.")
                return redirect("home")
        else:
            messages.error(request, "Such a directory doesn't exist. Please check your path.")
            return redirect("home")
    return render(request, "home.html", {})

def index(request):
    global directory, current_image_index, images, external_tags
    if not images:
        messages.error(request, "No images found. Please load a directory first.")
        return redirect('home')
    image_path = os.path.join(directory, images[current_image_index])
    # print(f"in index, image_path : {image_path}")
    ai_tags = generate_tags_for_image(image_path)
    # print(current_image_index)
    external_tags = dict(sorted(external_tags.items(), key=lambda item : item[1], reverse=True))
    external_tags_list = list(external_tags.keys())
    if len(external_tags_list) > 9:
        external_tags_list = external_tags_list[:10]
    # print(f"external tags dict : {external_tags}")
    existing_tags, existing_species, existing_reference = get_existing_info(image_path)
    print(f"existing info : {existing_tags}, {existing_species}, {existing_reference}")
    return render(request, 'index.html', {
        'image': image_path,
        'directory': directory,
        'ai_tags': ai_tags,
        'external_tags': external_tags_list,
        'existing_tags': existing_tags,
        'existing_species' : existing_species,
        'existing_reference': existing_reference
    })

def uploaded_file(request, filename):
    global directory
    return FileResponse(open(os.path.join(directory, filename), 'rb'))

@csrf_protect
def tag_image(request):
    global directory
    if request.method == 'POST':
        tags = request.POST.get('tags')
        image_path = request.POST.get('image_path')
        store_option = request.POST.get('store_option')
        species = request.POST.get('species')
        reference = request.POST.get('reference')
        absolute_image_path = os.path.join(directory, image_path)
        # print(f"absolute_image_path : {absolute_image_path}")
        # print(f"Received tags: {tags}, store option: {store_option}, for image: {absolute_image_path}")

        if store_option == 'metadata':
            save_metadata(absolute_image_path, tags, species, reference)

        print(f"tags : {tags}, species : {species}, reference : {reference}")

        save_info_to_text_file(absolute_image_path, tags, species, reference)
        save_info_to_excel(absolute_image_path, tags, species, reference)
        save_tags_to_json(absolute_image_path, tags)

    return redirect('index')

def prev_image(request, image):
    global current_image_index
    current_image_index = (current_image_index - 1) % len(images) if images else 0
    if current_image_index < 0:
        current_image_index = len(images) - 1
    return HttpResponseRedirect(reverse('index'))

def next_image(request, image):
    global current_image_index
    current_image_index = (current_image_index + 1) % len(images) if images else 0
    if current_image_index > len(images) - 1:
        current_image_index = 0
    return HttpResponseRedirect(reverse('index'))

def explore(request):
    assigned_tags = []
    unassigned_tags = []
    print('in explore')
    assigned_tags_str = ''
    if request.method == 'POST':
        print(request.POST)
        file = request.POST['directory']
        print(f'file : {file}')
        if os.path.isdir(file):
            # print('in explore if')
            file = None
            messages.error(request, "Please enter file path not folder")
            return render(request, 'explore.html',{
                'image': file,
                'assigned_tags': assigned_tags_str,
                'existing_tags': assigned_tags_str,
            })
        elif os.path.exists(file):
            assigned_tags_str, unassigned_tags = get_tags_for_image(file)
            ai_tags = generate_tags_for_image(file)
            return render(request, 'explore.html', {
                'image': file,
                'ai_tags' : ai_tags,
                'directory' : os.path.dirname(file),
                'assigned_tags': assigned_tags_str,
                'existing_tags': assigned_tags_str
            })
        else:
            # print('in explore else')
            file = None
            messages.error(request, "File not found. Please check the complete path of your image.")
            return render(request, 'explore.html',{
                'image': file,
                'assigned_tags': assigned_tags_str,
                'existing_tags': assigned_tags_str,
            })
    return render(request, 'explore.html', {'assigned_tags': assigned_tags_str, 'unassigned_tags': unassigned_tags})

def get_tags_for_image(image_path):
    """
    retrieve tags for particular image
    """
    assigned_tags, assigned_tags_str = [], ""
    unassigned_tags = [tag for tag in list(external_tags.keys()) if tag not in assigned_tags]
    
    for i in range(len(df)):
        if image_path == df.iloc[i][0]:
            print('found')
            assigned_tags = df.iloc[i][1]
            break
        
    if len(assigned_tags) > 0:
        if type(assigned_tags) == 'list':
            for i in range(len(assigned_tags) - 1):
                assigned_tags_str += assigned_tags[i] + ','
            assigned_tags_str += assigned_tags[-1]
        else:
            assigned_tags_str = assigned_tags

    return assigned_tags_str, unassigned_tags
