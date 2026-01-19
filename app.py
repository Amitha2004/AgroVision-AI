import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===============================
# PAGE CONFIG (APP NAME)
# ===============================
st.set_page_config(
    page_title="AgroVision AI | Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ===============================
# CUSTOM CSS (PRO UI)
# ===============================
st.markdown("""
<style>
.main { background-color: #0e1117; }
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background-color: #0d6efd;
    color: white;
    font-weight: 600;
}
.footer {
    color: #9aa0a6;
    text-align: center;
    font-size: 14px;
}
.app-name {
    font-size: 36px;
    font-weight: 700;
    color: #2ecc71;
}
.tagline {
    font-size: 16px;
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style="text-align:center;">
    <div class="app-name">🌿 AgroVision AI</div>
    <div class="tagline">
        Smart Leaf Disease Detection & Treatment System using Deep Learning
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("## 🌱 AgroVision AI")
    st.caption("AI-powered agricultural decision support")

    st.markdown("---")
    st.header("📌 How to Use")
    st.markdown("""
    1️⃣ Upload a **clear leaf image**  
    2️⃣ Ensure **good lighting**  
    3️⃣ System predicts the disease  
    4️⃣ View **confidence & treatment**
    """)

    st.markdown("---")
    st.warning(
        "⚠ This system provides **decision support only** "
        "and is not a replacement for professional agricultural advice."
    )

# ===============================
# DEVICE
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# CLASS NAMES
# ===============================
class_names = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

NUM_CLASSES = len(class_names)

# ===============================
# ELABORATED TREATMENT INFO
# ===============================
treatment_info = {

"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
("""• Grow resistant maize hybrids to reduce disease severity
• Apply fungicides such as Mancozeb or Chlorothalonil at early stages
• Avoid overhead irrigation to limit leaf wetness
• Practice crop rotation to break the disease cycle""",
"https://farmonaut.com/precision-farming/organic-vs-chemical-controlling-gray-leaf-spot-disease-in-corn-fields"),

"Corn_(maize)___Common_rust_":
("""• Use rust-resistant maize varieties
• Monitor fields regularly during humid conditions
• Apply fungicides if disease spreads rapidly
• Remove crop residues after harvest""",
"https://www.corteva.com/uk/tools-and-advice/disease-management/common-rust.html"),

"Corn_(maize)___Northern_Leaf_Blight":
("""• Select resistant or tolerant hybrids
• Practice crop rotation with non-host crops
• Apply fungicides during early disease development
• Ensure proper plant spacing for airflow""",
"https://www.corteva.com/uk/tools-and-advice/disease-management/northern-leaf-blight.html"),

"Corn_(maize)___healthy":
("""• Crop is healthy
• Maintain balanced fertilization
• Ensure proper irrigation and weed management
• Continue regular monitoring""",
"https://www.agriculture.com/crops/corn"),

"Grape___Black_rot":
("""• Remove and destroy infected berries and vines
• Apply fungicides such as Myclobutanil at flowering stage
• Maintain good air circulation through pruning
• Avoid prolonged leaf wetness""",
"https://www.youtube.com/watch?v=agIwEBm7Zao"),

"Grape___Esca_(Black_Measles)":
("""• Prune infected wood during dry seasons
• Avoid excessive pruning wounds
• Improve soil drainage and vine nutrition
• Remove severely infected vines""",
"https://www.msbiotech.net/en/mal-desca-della-vite-cure-e-trattamenti/"),

"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
("""• Apply copper-based fungicides preventively
• Remove infected leaves from vineyard
• Improve spacing and air circulation
• Avoid overhead irrigation""",
"https://plantvillage.psu.edu/topics/grape/infos"),

"Grape___healthy":
("""• Plant is healthy
• Maintain pruning schedule
• Ensure balanced nutrients and irrigation
• Monitor regularly for early disease signs""",
"https://www.youtube.com/watch?v=ner6ETMo5-0"),

"Pepper,_bell___Bacterial_spot":
("""• Use certified disease-free seeds
• Apply copper-based bactericides
• Avoid working in wet fields
• Practice crop rotation""",
"https://ipm.cahnr.uconn.edu/managing-bacterial-leaf-spot/"),

"Pepper,_bell___healthy":
("""• Plant is healthy
• Maintain soil fertility
• Provide adequate sunlight and water
• Continue regular inspection""",
"https://plantvillage.psu.edu/topics/pepper-bell/infos"),

"Tomato___Bacterial_spot":
("""• Use disease-free seeds or seedlings
• Apply copper-based sprays
• Avoid overhead watering
• Remove infected plant debris""",
"https://hort.extension.wisc.edu/articles/bacterial-spot-of-tomato/"),

"Tomato___Early_blight":
("""• Remove infected leaves immediately
• Apply fungicides like Chlorothalonil
• Practice crop rotation
• Avoid wet foliage""",
"https://www.youtube.com/watch?v=Lf6LrtuqFm8"),

"Tomato___Late_blight":
("""• Remove and destroy infected plants
• Apply fungicides such as Metalaxyl
• Avoid excess irrigation
• Ensure good air circulation""",
"https://www.youtube.com/watch?v=klaeUwprBzQ"),

"Tomato___Leaf_Mold":
("""• Reduce humidity in greenhouse conditions
• Improve ventilation
• Apply fungicides if severe
• Remove infected leaves""",
"https://www.youtube.com/watch?v=oEmY2aHUuoA"),

"Tomato___Septoria_leaf_spot":
("""• Remove infected foliage promptly
• Apply recommended fungicides
• Avoid splashing water on leaves
• Practice crop rotation""",
"https://www.youtube.com/watch?v=bI0B4IsQT3w"),

"Tomato___Spider_mites Two-spotted_spider_mite":
("""• Spray neem oil or insecticidal soap
• Increase humidity to reduce mite population
• Introduce natural predators if possible
• Avoid excessive pesticide use""",
"https://www.youtube.com/watch?v=TNMoLhT2A14"),

"Tomato___Target_Spot":
("""• Apply fungicides at early disease stages
• Improve air circulation
• Remove infected leaves
• Avoid prolonged leaf wetness""",
"https://plantix.net/en/library/plant-diseases/300050/"),

"Tomato___Tomato_Yellow_Leaf_Curl_Virus":
("""• Control whitefly population
• Remove infected plants immediately
• Use virus-resistant varieties
• Maintain field hygiene""",
"https://www.youtube.com/watch?v=D-58aAFIyCQ"),

"Tomato___healthy":
("""• Plant is healthy
• Maintain balanced fertilization
• Provide proper irrigation
• Continue monitoring""",
"https://www.webmd.com/food-recipes/ss/slideshow-tomato-health-benefits")
}

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    left, right = st.columns([1, 1])

    with left:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=350)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1)[0]

    top3 = torch.topk(probs, 3)
    pred_idx = top3.indices[0].item()
    confidence = top3.values[0].item()
    disease = class_names[pred_idx]

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Prediction")
        st.markdown(f'<span class="badge">{disease}</span>', unsafe_allow_html=True)

        st.markdown("### 📊 Confidence")
        st.progress(int(confidence * 100))
        st.write(f"**{confidence:.2%}**")

        if confidence < 0.6:
            st.warning("Low confidence. Please upload a clearer image.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔝 Top-3 Predictions")
    for i in range(3):
        st.write(f"{i+1}. {class_names[top3.indices[i]]} — {top3.values[i]:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Treatment Recommendation")
    st.write(treatment_info[disease][0])
    st.markdown(f"🔗 [Learn more]({treatment_info[disease][1]})")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown(
    '<hr><p class="footer">🌿 AgroVision AI | Smart Leaf Disease Detection & Treatment System</p>',
    unsafe_allow_html=True
)
