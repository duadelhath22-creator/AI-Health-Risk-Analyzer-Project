import streamlit as st
import seaborn as sns
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="AI Health Risk Analyzer",
    page_icon="🏥",
    layout="wide"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

div[data-testid="stMetric"]{
    background-color:#EDF5EA;
    border-radius:15px;
    padding:18px;
    border:1px solid #D5E8D4;
}

/* Hover Animation */
div[data-testid="stMetric"]:hover{
    transform:translateY(-4px);
    transition:0.3s;
    box-shadow:0 6px 20px rgba(0,0,0,0.15);
}

/* Buttons */
.stButton>button{
    border-radius:25px;
    height:50px;
    background:#8AAE92;
    color:white;
    font-weight:bold;
}

.stButton>button:hover{
    background:#507D5C;
    transition:0.3s;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#EDF5EA;
}

/* Page Animation */
.main{
    animation:fadeIn 0.7s ease-in;
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(15px);
    }
    to{
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATASET ----------------
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters
# ---------------- LOAD MODEL ----------------
model = joblib.load("models/model.pkl")
y_pred = model.predict(X_test)
# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Health Risk Analyzer",
    page_icon="🏥",
    layout="wide"
)

# ---------------- NAVIGATION ----------------
st.sidebar.title("🏥 AI Health Risk Analyzer")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Prediction",
        "📊 Dataset Analysis",
        "📈 Model Performance",
        "🤖 Clustering",
        "ℹ️ About"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.title("🏥 AI Health Risk Analyzer")
    
    st.caption(
    "Early Breast Cancer Detection using Machine Learning"
)

    st.markdown("---")

    st.header("Welcome!")

    st.write("""
    This application predicts whether a breast tumor is **Benign** or **Malignant**
    using a trained Machine Learning model.

    Use the navigation menu to explore different pages.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Project Information")

        st.write("""
        - Model : Random Forest Classifier
        - Dataset : Breast Cancer Wisconsin Dataset
        - Features : 30
        - Accuracy : 96.49%
        - Framework : Streamlit
        """)

    with col2:
        st.subheader("🤖 Features")

        st.write("""
        ✔ Breast Cancer Prediction

        ✔ Dataset Analysis

        ✔ Model Performance

        ✔ Clustering (K-Means)

        ✔ Association Rules

        ✔ Professional Dashboard
        """)

    st.markdown("---")

    st.success("Select a page from the navigation menu.")

# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "🔬 Prediction":

    st.title("🔬 Breast Cancer Prediction")

    st.write("Enter all the patient's measurements below.")
    

    feature_names = list(data.feature_names)

    user_input = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        if i % 2 == 0:
            with col1:
                value = st.number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        format="%.5f"
            )
        else:
            with col2:
                value = st.number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        format="%.5f"
                )
        user_input.append(value)

    if st.button("🔍 Predict"):

        input_df = pd.DataFrame([user_input], columns=feature_names)

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Benign Tumor")
        else:
            st.error("⚠️ Malignant Tumor")

        st.subheader("Prediction Confidence")

        benign = probability[0][1]
        malignant = probability[0][0]

        st.write(f"Benign Probability : **{benign*100:.2f}%**")
        st.progress(float(benign))

        st.write(f"Malignant Probability : **{malignant*100:.2f}%**")
        st.progress(float(malignant))

# ==================================================
# DATASET ANALYSIS PAGE
# ==================================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    rows, cols = df.shape

    st.write("Rows :", rows)
    st.write("Columns :", cols)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Class Distribution")

    class_counts = df["target"].value_counts()

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["Malignant", "Benign"],
        class_counts.values
    )

    ax.set_ylabel("Number of Patients")
    ax.set_title("Class Distribution")

    st.pyplot(fig)

    st.subheader("Feature Histogram")

    feature = st.selectbox(
        "Select Feature",
        data.feature_names
    )

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(df[feature], bins=20)

    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,10))
    numeric_df = df.drop(columns=["target"])
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
    
elif page == "📈 Model Performance":  
    st.title("📈 Model Performance") 
    st.subheader("Model Accuracy")

    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy : {accuracy*100:.2f}%")
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    st.subheader("Classification Report")

    report = classification_report(
      y_test,
      y_pred,
      output_dict=True
)

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)
    st.subheader("Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
      "Feature": data.feature_names,
      "Importance": importance
})

    importance_df = importance_df.sort_values(
      by="Importance",
      ascending=False
)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
      importance_df["Feature"],
      importance_df["Importance"]
)

    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")

    plt.gca().invert_yaxis()

    st.pyplot(fig)
elif page=="🤖 Clustering":
    st.title("🤖 K-Means Clsutering")
    st.subheader("Dataset with Cluster labels")
    st.dataframe(df.head())
    st.subheader("Cluster Information")
    st.write("Number of Clusters:", kmeans.n_clusters)
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
    df["mean radius"],
    df["mean texture"],
    c=df["Cluster"]
)

    ax.set_xlabel("Mean Radius")
    ax.set_ylabel("Mean Texture")
    ax.set_title("K-Means Clustering")

    st.pyplot(fig)
    st.subheader("Cluster Distribution")
    cluster_counts = df["Cluster"].value_counts()
    st.write(cluster_counts)
    st.subheader("Cluster Count Chart")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Patients") 
    ax.set_title("Patients in each Cluster")
    st.pyplot(fig)
    st.subheader("Cluster Centers")

    centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
)

    st.dataframe(centers)
elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("---")

    st.header("🏥 AI Health Risk Analyzer")

    st.write("""
    The **AI Health Risk Analyzer** is a Machine Learning-based web application
    developed to predict whether a breast tumor is **Benign** or **Malignant**
    using patient medical measurements.

    The application also provides dataset analysis, model performance evaluation,
    and clustering analysis to better understand the data and the trained model.
    """)
    st.subheader("🎯 Project Objective")

    st.write("""
    The objective of this project is to develop an intelligent healthcare
    application that assists in the early prediction of breast cancer using
    Machine Learning techniques. The system helps demonstrate how AI can support
    medical diagnosis through data-driven predictions.
    """)
    st.subheader("📊 Dataset")

    st.write("""
    - Dataset : Breast Cancer Wisconsin Dataset
    - Total Records : 569
    - Total Features : 30
    - Target Classes :
    - Benign
    - Malignant
    - Source : Scikit-learn
    """)
    st.subheader("🛠 Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Joblib
    """)
    st.subheader("🤖 Machine Learning Techniques")

    st.write("""
    - Random Forest Classifier
    - Data Analysis
    - Model Performance Evaluation
    - K-Means Clustering
    """)
    st.subheader("✨ Application Features")

    st.write("""
    - Breast Cancer Prediction

    - Dataset Analysis

    - Model Performance Evaluation

    - K-Means Clustering

    - Interactive Dashboard
    """)
    st.subheader("👩‍💻 Developer")

    st.write("""
    **Name:** Dua

    **Project:** AI Health Risk Analyzer

    **Department:** Robotics and AI

    **College:** NMAM Institute of Technology
    """)
    st.markdown("---")

    st.success("Thank you for exploring the AI Health Risk Analyzer!")
    