import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from io import StringIO

st.markdown(
    f"""
       <style>
       .e1f1d6gn0{{
        background-color: #0000cc;      
        border-radius:5px;
        display:flex;
        flex-direction:column;
        justify-content:space-around;
        padding:20px;
        }}      
        </style>
       """,
    unsafe_allow_html=True
)
st.markdown("""<h1 classname="name-title" style="color:White;text-align: center;">Profanity Dection<h1/>""",unsafe_allow_html=True)
tab1, tab2, tab3,tab4 = st.tabs(["Generate", "Load", "Train","Predict"])
user_upload_file_path = ''
with tab1:
   st.markdown("""<h2 style="color:White;">Generate Page<h2/>""",unsafe_allow_html=True)
   st.markdown("""<p style="color:White;">กรุณากรอกชื่อไฟล์<p/>""",unsafe_allow_html=True)
   title = st.text_input('', '',label_visibility="collapsed")
   if st.button("Create"):
      file_name=f"data/{title}.csv"
      file = open(f"{file_name}", "w")
      with open(file_name, 'a', encoding='utf-8') as f:
         row = "text,label\nพ่อมึงตาย,1\nสวัสดี,0"
         f.write(row)
      st.markdown(f"""<p style="color:White">สร้างไฟล์ {title} สำเร็จ<p/>""",unsafe_allow_html=True)

with tab2:
   st.markdown("""<h2 style="color:White;">Load Page<h2/>""",unsafe_allow_html=True)
   uploaded_file = st.file_uploader("Choose a CSV file")
   if uploaded_file is not None:
      st.markdown("""<p style="color:White;">โหลดสำเร็จ<p/>""",unsafe_allow_html=True)
      user_upload_file_path = f"data/{uploaded_file.name}"
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)


with tab3:
   st.markdown("""<h2 style="color:White;">Train Page<h2/>""",unsafe_allow_html=True)
   word = st.text_input('ใส่ประโยคที่ต้องการ', '')
   profanity = st.radio(
        "",
        ["ไม่สุภาพ","สุภาพ"],
        label_visibility="collapsed",
        horizontal=True,
    )
   if st.button("Train"):
      with open(user_upload_file_path, 'a', encoding='utf-8') as f:
         profanity_v = "0" if profanity == "สุภาพ" else '1'
         row = f"\n{word},{profanity_v}"
         f.write(row)
      
      df = pd.read_csv(user_upload_file_path)
      train_size = int(0.8 * len(df))
      train_data = df[:train_size]
      test_data = df[train_size:]
      vectorizer = CountVectorizer()
      X_train = vectorizer.fit_transform(train_data['text'])
      X_test = vectorizer.transform(test_data['text'])
      y_train = train_data['label']
      y_test = test_data['label']
      clf = SVC(kernel='linear')
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      st.markdown("""<p style="color:White;">ฝึกสำเร็จ<p/>""",unsafe_allow_html=True)
      dataframe = pd.read_csv(user_upload_file_path)
      st.write("Accuracy:", accuracy_score(y_test, y_pred))
      st.write("Precision:", precision_score(y_test, y_pred))
      st.write("Recall:", recall_score(y_test, y_pred))
      st.write("F1-score:", f1_score(y_test, y_pred))
      st.write(dataframe)

with tab4:
   st.header("Predict")
   predict_input = st.text_input('ใส่ประโยคที่ต้องการ', '',key="Predict")
   if st.button("Predict"):
      df = pd.read_csv(user_upload_file_path)
      vectorizer = CountVectorizer()
      X = vectorizer.fit_transform(df['text'])
      y = df['label']
      clf = SVC(kernel='linear')
      clf.fit(X, y)
      user_input_vector = vectorizer.transform([predict_input])
      prediction = clf.predict(user_input_vector)

      if prediction == 1:
         st.markdown(f"""<p style="color:White;">{predict_input} เป็น: ไม่สุภาพ<p/>""",unsafe_allow_html=True)

      else:
         st.markdown(f"""<p style="color:White;">{predict_input} เป็น: สุภาพ"<p/>""",unsafe_allow_html=True)