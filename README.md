# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: SANJANA M NAIK

INTERN ID: CT04DH171

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH


## 📝 Project Overview 


As part of my internship at CodTech, I successfully completed Task 3: End-to-End Data Science Project, which involved developing and deploying an image classification model to distinguish between cats and dogs. This project was designed to demonstrate a complete data science workflow—from organizing the dataset and training the deep learning model to deploying it in a web application for real-world interaction. The main objective of this task was to create an end-to-end machine learning pipeline that could classify uploaded images as either a cat or a dog with a simple and interactive user interface. The dataset was manually created and organized under a dataset/train/ directory, which included two subfolders—cats/ and dogs/, each containing four sample images (e.g., c1.jpeg to c4.jpeg for cats and d1.jpeg to d4.jpeg for dogs). I used Python 3.13 as the programming language, and PyTorch as the core deep learning framework for building and training a simple Convolutional Neural Network (CNN). Additional Python libraries such as torchvision were used for image transformations, and NumPy and Matplotlib supported array operations and visualizations. Model training was handled in the train_model.py script, which generated a trained model saved as cat_dog_model.pth inside the models/ directory. This saved model was later used for inference.

The application was deployed using Flask, a lightweight and flexible Python web framework. The file main.py handled image classification logic by loading the trained model and making predictions, while app.py served as the entry point for launching the web server. The front-end was created using HTML and placed inside the templates/index.html file. Users can interact with the application by uploading images through this UI, after which the backend processes the input, performs inference, and displays whether the image is a cat or a dog. Uploaded images are temporarily stored in the static/uploads/ directory for processing. All development was carried out using Visual Studio Code (VS Code) on a Windows environment. The model achieved around 75% accuracy, which is reasonable for such a small dataset and was validated through both console output and test image predictions.


<img width="433" height="757" alt="Image" src="https://github.com/user-attachments/assets/25be05da-545d-4371-9bd0-c7ce528660eb" />

<img width="450" height="573" alt="Image" src="https://github.com/user-attachments/assets/b2c14d39-6006-4d8d-a84e-a223d8b09025" />
