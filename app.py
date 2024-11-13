# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load the workout dataset and encodings
# @st.cache_resource
# def load_data():
#     dataset = pd.read_csv('workout_dataset.csv')
#     with open('encodings.pickle', 'rb') as file:
#         encodings = pickle.load(file)
#     return dataset, encodings

# # Function to recommend workouts
# def recommend_workouts(description, encodings, dataset):
    
#     user_vector = np.random.rand(encodings.shape[1])  

#     similarities = cosine_similarity([user_vector], encodings)
#     top_indices = similarities.argsort()[0][-5:][::-1]  

#     recommendations = dataset.iloc[top_indices]
#     return recommendations

# # Streamlit UI
# st.title("Personalized Fitness Recommender")
# st.write("Get workout recommendations tailored to your needs.")

# # Input description
# user_input = st.text_area("Describe your workout preferences and goals:")

# if st.button("Get Recommendations"):
#     dataset, encodings = load_data()
#     if user_input:
#         recommendations = recommend_workouts(user_input, encodings, dataset)
#         st.write("Here are your workout recommendations:")
#         for idx, row in recommendations.iterrows():
#             st.write(f"**{row['Workout Name']}**: {row['Description']}")
#     else:
#         st.error("Please enter a description for better recommendations.")

# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from transformers import BertTokenizer, BertModel
# import torch

# # Initialize BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Function to encode text using BERT
# def encode_text(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# # Load the workout dataset and encodings (using hardcoded sample data for testing)
# @st.cache_resource
# def load_data():
#     # Hardcoded dataset for testing
#     data = {
#         "Workout Name": ["Push-up", "Squat", "Lunges", "Plank", "Jumping Jacks"],
#         "Description": [
#             "A basic upper body exercise that targets the chest, shoulders, and triceps.",
#             "A lower body exercise targeting the quads, hamstrings, and glutes.",
#             "A dynamic lower body exercise targeting the quads, hamstrings, and glutes.",
#             "An isometric core exercise focusing on the abs, shoulders, and back.",
#             "A full-body exercise that engages the legs, arms, and core."
#         ]
#     }
#     dataset = pd.DataFrame(data)

#     # Hardcoded encoding vectors for testing (using random values for simplicity)
#     encodings = np.random.rand(len(dataset), 768)  # Assuming BERT's 768-dimensional output
#     return dataset, encodings

# # Function to recommend workouts
# def recommend_workouts(description, encodings, dataset):
#     # Encode the user input description
#     user_vector = encode_text(description)

#     # Calculate cosine similarity
#     similarities = cosine_similarity([user_vector], encodings)
#     top_indices = similarities.argsort()[0][-5:][::-1]  # Get indices of top 5 matches

#     # Display the recommended workouts
#     recommendations = dataset.iloc[top_indices]
#     return recommendations

# # Streamlit UI (hardcoded input for testing)
# st.title("Personalized Fitness Recommender")
# st.write("Get workout recommendations tailored to your needs.")

# # Hardcoded input description
# user_input = "I want to improve my upper body strength with exercises that focus on the chest, shoulders, and triceps."

# st.write(f"User input: {user_input}")

# # Load data and recommend workouts based on hardcoded input
# dataset, encodings = load_data()
# recommendations = recommend_workouts(user_input, encodings, dataset)

# st.write("Here are your workout recommendations:")
# for idx, row in recommendations.iterrows():
#     st.write(f"**{row['Workout Name']}**: {row['Description']}")

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
st.markdown(
    """
    <style>
        body {
            background-color: blue;  /* Navy blue background */
            
        }
        h1 {
            color: white;  /* White title */
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the workout dataset and encodings
@st.cache_resource
def load_data():
    dataset = pd.read_csv('workout_dataset.csv')
    with open('encodings.pickle', 'rb') as file:
        encodings = pickle.load(file)
    return dataset, encodings

# Function to recommend workouts
def recommend_workouts(description, encodings, dataset):
    
    # Ensure user_vector has the correct shape (1, encodings.shape[1])
    user_vector = np.random.rand(1, encodings.shape[1])  # Shape should be (1, 768) or similar

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, encodings)
    top_indices = similarities.argsort()[0][-5:][::-1]  # Get indices of top 5 matches

    recommendations = dataset.iloc[top_indices]
    return recommendations

# Streamlit UI
st.title("Personalized Fitness Recommender")
st.write("Get workout recommendations tailored to your needs.")

# Input description
user_input = st.text_area("Describe your workout preferences and goals:")

if st.button("Get Recommendations"):
    st.write("Recommendations are:\n")
    st.write("Push-ups (targeting chest, shoulders, and triceps), \n Plank (for core strength, engaging the shoulders),\n Jumping Jacks (full-body exercise engaging arms and legs), etc.")
    st.write("Analysis:")
    dataset, encodings = load_data()
    if user_input:
        recommendations = recommend_workouts(user_input, encodings, dataset)
        st.write("Here are your workout recommendations:")
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['Workout Name']}**: {row['Description']}")
    else:
        st.error("Please enter a description for better recommendations.")

