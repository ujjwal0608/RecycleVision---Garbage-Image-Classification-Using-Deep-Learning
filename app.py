import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
# From Created File
from model import ResNet
from class_names import class_names

# --------------------Page Config--------------------
st.set_page_config(
    page_title = 'Garbage Classifier',
    page_icon = '♻️',
    layout = 'wide'
)

# --------------------Custom CSS--------------------
st.markdown('''
<style>

.main{
  background-color: #07111f;
  color = white;
}

h1, h2, h3, h4, h5, h6{
  color: white !important;
}

.stButton>button{
  background-color: transparent;
  color: white;
  border: 2px solid #ff4b4b;
  border-radius: 10px;
  padding: 10px 20px;
  font-weight: bold;
}

.stButton>button:hover{
  background-color: #ff4b4b;
  color: white;
}

.card {
  background-color: #10243b;
  padding: 20px;
  border-radius: 15px;
  margin-bottom: 20px;
}

.result-card {
  background-color: #124d2e;
  padding: 20px;
  border-radius: 15px;
  font-size: 24px;
  font-weight: bold;
  text-align: center;
}

.info-card {
  background-color: #124d2e;
  padding: 15px;
  border-radius: 15px;
  margin-bottom: 15px;
}

</style>
''', unsafe_allow_html=True)

# To GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the Model We Created
model = ResNet(num_classes=len(class_names))

checkpoint = torch.load(
    "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.to(device)
model.eval()

# Image Transformation
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
])

# Defining a Prediction Function
def predict_image(image):

  image = transformation(image).unsqueeze(0).to(device)

  with torch.no_grad():

    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

  predicted_class = class_names[predicted.item()]
  confidence_score = confidence.item()*100

  return predicted_class, confidence_score


# User Interface StreamLit
st.title('♻️ Garbage Image Classification')

st.write(
    'Upload a waste image to identify its type and understand its environmental impact.'
)

# Uploading a Image
uploaded_file = st.file_uploader(
    'Upload an image (JPG, PNG, JPEG)',
    type=["jpg", "jpeg", "png"]
)

# Classify Button
if uploaded_file is not None:

  image = Image.open(uploaded_file)
  image = image.convert('RGB')
  classify = st.button("Classify")

  if classify:
    prediction, confidence = predict_image(image)
    # 3 Column Layout
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
      st.markdown(
          '<div class = "card"><h3>Upload Image</h3></div>',
          unsafe_allow_html=True
      )
      st.image(image, caption='Uploaded Image', use_container_width = True,)

    with col2:
      st.markdown(
          '<div class="card"><h3>Classification Result</h3></div>',
          unsafe_allow_html = True
      )
      st.markdown(
          f'''
          <div class="result-card">
            The Image is Predicted as: {prediction}
            <br><br>
            The Confidence is: {confidence:.2f}%
          </div>
      ''',unsafe_allow_html=True
          )

    with col3:
      st.markdown(
          '<div class="card"><h3>Environmental Awareness</h3></div>',
          unsafe_allow_html = True
      )
      if prediction.lower() == "metal":
        st.markdown(
            '''
            <div class = 'info-card'>
            ♻️ Metal waste can be recycled efficiently.
            </div>
            ''',
            unsafe_allow_html = True
        )

        st.markdown(
            '''
            <div class="info-card">
            Aluminum recycling saves huge energy.
            </div>
            ''',
            unsafe_allow_html = True
        )

        st.markdown(
            '''
            <div class="info-card">
            Recycling metals reduces CO₂ emissions.
            </div>
            ''',
            unsafe_allow_html=True
        )

      elif prediction.lower() == 'plastic':
        st.markdown(
            '''
            <div class="info-card">
            ♻️ Plastic waste harms oceans and animals.
            </div>
            ''',
            unsafe_allow_html = True
        )

        st.markdown(
            '''
            <div class="info-card">
            Recycle plastics to reduce pollution.
            </div>
            ''',
            unsafe_allow_html=True
        )

      elif prediction.lower() == 'paper':
        st.markdown(
            '''
            <div class="info-card">
            🌳 Recycling paper saves trees and water.
            </div>
            ''',
            unsafe_allow_html = True
        )
      else:
        st.markdown(
            '''
            <div class="info-card">
            Proper waste segregation helps the environment.
            </div>
            ''',
            unsafe_allow_html = True
        )
    st.success(f"The image is classified as {prediction}")
