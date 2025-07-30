from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model import CustomCNN
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = CustomCNN()
model.load_state_dict(torch.load('models/cat_dog_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match training input
    transforms.ToTensor()
])


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(probs, 1)
                class_names = ['cat', 'dog']
                prediction = class_names[predicted.item()]
                image_url = filepath

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
