import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import av
import time
import pandas as pd 
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from tensorflow.keras.utils import img_to_array
from streamlit_extras.stodo import to_do
from matplotlib import rcParams
import openai

openai.api_key = "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"

rcParams['font.family'] = 'serif'


# Emotion labels
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
detected_emotions = []
detection_times = []
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
emotion_df = pd.DataFrame(columns=["Emotion", "Time"])

new_df=pd.DataFrame(columns=['Emotion', 'Time'])
st.session_state.my_variable = new_df
# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier(r"C:\Users\sarra\my_mental_health\haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")
#load my model    
my_mental_health = load_model(r"C:\Users\sarra\my_mental_health\my_mental_health2.h5")    
    




class Face_emotion(VideoTransformerBase):
    def __init__(self):
        self.emotion_df = pd.DataFrame(columns=["Emotion", "Time"])

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = my_mental_health.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                emotion = str(finalout)
                detection_time = time.strftime('%Y-%m-%d %H:%M:%S')
                self.emotion_df.loc[len(self.emotion_df)] = [emotion, detection_time]

            label_position = (x, y)
            cv2.putText(img, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

 
#upliftting message function
def generate_uplifting_message(emotion):
    response = openai.Completion.create(
        prompt=f"can you write an uplifting small simple one-sentence message for someone who is feeling {emotion}?with an emoji",
        engine="text-davinci-003",
        temperature=0.8,
        max_tokens=100,
        n=1,
        stop=None,
        echo=False
    )

    message = response.choices[0].text.strip()
    return message




#function for plots
def further_analysis(emotion_df):
    #1-plot the bar chart for the detected emotions between start and stop of the webcam
    def display_bar_plot():
        title1 = "Your emotions Today"
        st.markdown(f'<h2 style="color: black;text-align:center;">{title1}</h2>', unsafe_allow_html=True)
        emotion_counts = emotion_df['Emotion'].value_counts()
        
        fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(emotion_counts.index, emotion_counts.values, color='palevioletred')
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Count')
        #ax.set_title('Detected Emotions')
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.set_facecolor('#FBF2F7')
        st.pyplot(fig)
        
        

    def display_pie_chart():
        # Load data from CSV
        data = pd.read_csv('emotion_data.csv')
        max_emotions = data.groupby("Emotion").count().reset_index()
        max_emotion = max_emotions.sort_values('Time', ascending=False)['Emotion'].iloc[1]

        first_date = data['Time'].min()
        last_date = data['Time'].max()

        title1 = "Emotion Distribution"
        title2 = f"from {first_date} to {last_date}"

        st.markdown(f'<h2 style="color: black;text-align:center;">{title1}</h2>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: black; text-align: center; font-size: 16px;">{title2}</h2>', unsafe_allow_html=True)

        # Create pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        emotion_counts = data['Emotion'].value_counts()
        explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', explode=explode,
               colors=['#D8BFD8', '#87CEFA', '#FF7F50', '#FFD700', '#98FB98', '#FF69B4'], radius=0.9, textprops={'fontsize': 12})
        fig.set_facecolor('#FBF2F7')

        # Display the pie chart in the first column
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig)

        # Display the message in the second column
        with col2:
            if max_emotion == 'sad':
                st.markdown('<div class="chat-bubble"><p>I\'ve noticed that you\'ve been feeling sad lately. '
                            'It\'s important to take care of your emotional well-being. Reach out to someone you trust, '
                            'engage in activities that uplift your mood, and remember that brighter days are ahead.</p></div>',
                            unsafe_allow_html=True)
            elif max_emotion == 'angry':
                st.markdown('<div class="chat-bubble"><p>I\'ve noticed that you\'ve been feeling angry recently. '
                            'It\'s natural to experience anger, but it\'s important to find healthy ways to manage it. '
                            'Take a moment to breathe deeply, practice relaxation techniques, and consider expressing your '
                            'feelings in a constructive manner.</p></div>', unsafe_allow_html=True)
            elif max_emotion == 'fear':
                st.markdown('<div class="chat-bubble"><p>I\'ve noticed that you\'ve been feeling fearful lately. '
                            'Remember that fear is a normal human emotion, but it\'s essential not to let it control you. '
                            'Take small steps to face your fears, seek support from loved ones, and focus on positive '
                            'aspects to build resilience.</p></div>', unsafe_allow_html=True)
            elif max_emotion == 'happy':
                st.markdown('<div class="chat-bubble"><p>I\'m glad to see that you\'ve been feeling happy! Embrace this '
                            'positive emotion and continue to engage in activities that bring you joy and fulfillment. '
                            'Spread your happiness to others and cherish these moments of positivity.</p></div>',
                            unsafe_allow_html=True)
            elif max_emotion == 'neutral':
                st.markdown('<div class="chat-bubble"><p>I see that you\'ve been feeling neutral recently. '
                            'It\'s alright to have moments of calm and neutrality. Take this time to reflect, find balance '
                            'in your life, and consider exploring new opportunities or hobbies to add more excitement.</p></div>',
                            unsafe_allow_html=True)
            elif max_emotion == 'surprise':
                st.markdown('<div class="chat-bubble"><p>You seem to be feeling surprised lately. '
                            'Embrace the unexpected and be open to new experiences. Enjoy the thrill and wonder that '
                            'surprises bring!</p></div>', unsafe_allow_html=True)

        # Apply CSS styles
        st.markdown(
            """
            <style>
            .chat-bubble {
                background-color: #D8BFD8;
                color: black;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
                display: inline-block;
                max-width: 80%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        
    col1, col2 = st.columns(2)

    # Button to display bar plot
    if col1.button("Your Emotions Today", key="bar_plot"):
        display_bar_plot()

    # Button to display pie chart
    if col2.button("Your Emotions Tracker", key="pie_chart"):
        display_pie_chart()



def main():

    st.markdown(
        """
        <style>
        .title-style {
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="title-style">🌸MY MENTAL HEALTH TRACKER🌸</h1>', unsafe_allow_html=True)

    # Rest of your code

# to do list 
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                       ["Live detection", "Track your emotions"])

    if app_mode == "Live detection":
        with st.sidebar:
            st.sidebar.header("How was your day?")
            to_do(
                [(st.write, "💗did you decide to be happy?")],
                "happy",
            )
            to_do(
                [(st.write, "💪did you tell your self how strong you are ?")],
                "strong",
            )
            to_do(
                [(st.write, "😄did you smile enough?")],
                "smile",
            )

        st.markdown(
            """
            <style>
            .header-style {
                color:black;
                background-color: transparent;
            }
            .chat-bubble {
                background-color: #D8BFD8;
                padding: 10px;
                margin: 10px;
                border-radius: 10px;
                display: inline-block;
            }
            .chat-bubble p {
                margin: 0;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True)

        #st.markdown('<h1 class="header-style">Webcam Live Feed</h1>', unsafe_allow_html=True)
#starting the webcam
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=Face_emotion)

        # html_analysis = """
        #     <div style="background-color:#FFF0F5;padding:10px">
        #         <h4 style="color:black;text-align:center;">
        #             How are you today?
        #         </h4>
        #     </div>
        #     </br>
        # """
        #st.markdown(html_analysis, unsafe_allow_html=True)

        click_me = st.button("I have something to tell you 👇")
        if webrtc_ctx.video_processor:
            while True:
                try:
                    time.sleep(1)
                    emotion_df = webrtc_ctx.video_transformer.emotion_df
                    emotion_df.to_csv('new_emotion.csv', index=False)
                    try:
                        existing_df = pd.read_csv('emotion_data.csv')
                    except FileNotFoundError:
                        existing_df = pd.DataFrame(columns=['Emotion', 'Time'])
                            
                    if click_me:
                        click_me = False  # Reset button state
                        updated_df = existing_df.append(emotion_df, ignore_index=True)
                        updated_df.to_csv('emotion_data.csv', index=False)

                        emotion_counts = emotion_df['Emotion'].value_counts()
                        most_prevalent_emotion = emotion_counts.index[0]
                        uplifting_message = generate_uplifting_message(most_prevalent_emotion)
                        st.markdown(f'<div class="chat-bubble"><p>{uplifting_message}     </p</div>',unsafe_allow_html=True)
                        
                except Exception as e:
                    break

        else:
            st.write("No emotions detected yet")
    if app_mode == "Track your emotions":
    
        further_analysis(pd.read_csv(r"C:\Users\sarra\new_emotion.csv"))
if __name__ == "__main__":
    main()
