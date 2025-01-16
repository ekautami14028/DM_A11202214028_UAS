import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import streamlit as st

from web_functions import train_model

def plot_confusion_matrix(y_test, y_pred):
    mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

def app(df, x, y):
    warnings.filterwarnings('ignore')

    st.title("Visualisasi Prediksi Penyakit Paru-Paru")

    if st.checkbox("Plot Confusion Matrix"):
        model, _ = train_model(x, y)  
        y_pred = model.predict(x)  
        plot_confusion_matrix(y, y_pred)

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['Tidak', 'Ya']
        )

        st.graphviz_chart(dot_data)
