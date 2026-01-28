
import streamlit as st
from fastai.vision.all import *


st.title("Chess Piece Classifier")
st.text("Hello, this is Renikha!")




def extract_chess_label(file_path):
   file_name_parts = str(file_path).split("/")
   folder_name = file_name_parts[-2]
   #print(folder_name)


   file_path_parts = folder_name.split("-")
   #print(file_path_parts)


   chess_label = ""


   for i in range(len(file_path_parts)):
       if i != len(file_path_parts) - 1:
           chess_label += file_path_parts[i] + "_"


   chess_label = chess_label[:-1]
   if chess_label == "":
      chess_label = "bishop"
   return chess_label


#print(extract_chess_label("/kaggle/input/chess-pieces-detection-images-dataset/Queen-Resized/00000000_resized.jpg"))
#print(extract_chess_label("/kaggle/input/chess-pieces-detection-images-dataset/knight-resize/00000006_resized.jpg"))




chess_name_model = load_learner("chess_name_model_fastai2_8_4.pkl")
#file = input("Enter your filename:")
uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
   piece_img = PILImage.create(uploaded_file)
   prediction = chess_name_model.predict(piece_img)
   print(prediction)
   label = prediction[0]
   if label == "":
      label = "Bishop"

   img_label = prediction[0]
   confidence_index = int(prediction[1])
   confidence = prediction[2][confidence_index]

   st.text(f"{img_label} , {confidence}% confidence.")
   st.image(uploaded_file, caption="UploadedImage", use_container_width=True)



# probabilities - a tensor of probabilities for each class
# The 3rd element is a torch.Tensor representing the predicted probabilities for each class, aligned with the data loader’s class order.


