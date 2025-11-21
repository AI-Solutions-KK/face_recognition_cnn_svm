import streamlit as st
from pathlib import Path

from app.inference import load_models, predict_image
from app.config import DATA_ROOT, IMAGE_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_IMAGE_PATH = REPORTS_DIR / "confusion_matrix_normalized.png"

# -------------------------------------------------------------------
# STATIC REPORT TEXTS (Block 8 and Block 10)
# -------------------------------------------------------------------
TRAINING_REPORT_TEXT = """                             precision    recall  f1-score   support

          pins_Adriana Lima       1.00      1.00      1.00        32
          pins_Alex Lawther       1.00      1.00      1.00        23
    pins_Alexandra Daddario       0.97      1.00      0.99        34
          pins_Alvaro Morte       1.00      1.00      1.00        30
           pins_Amanda Crew       1.00      1.00      1.00        30
          pins_Andy Samberg       1.00      1.00      1.00        29
         pins_Anne Hathaway       0.97      1.00      0.98        30
        pins_Anthony Mackie       1.00      1.00      1.00        30
         pins_Avril Lavigne       1.00      0.96      0.98        24
           pins_Ben Affleck       1.00      1.00      1.00        30
            pins_Bill Gates       1.00      1.00      1.00        30
          pins_Bobby Morley       0.97      1.00      0.98        30
      pins_Brenton Thwaites       0.97      0.97      0.97        31
        pins_Brian J. Smith       1.00      1.00      1.00        30
           pins_Brie Larson       1.00      0.92      0.96        25
           pins_Chris Evans       1.00      1.00      1.00        25
       pins_Chris Hemsworth       1.00      1.00      1.00        24
           pins_Chris Pratt       1.00      1.00      1.00        26
        pins_Christian Bale       1.00      1.00      1.00        23
     pins_Cristiano Ronaldo       1.00      1.00      1.00        30
    pins_Danielle Panabaker       0.96      1.00      0.98        27
       pins_Dominic Purcell       1.00      1.00      1.00        30
        pins_Dwayne Johnson       1.00      1.00      1.00        30
          pins_Eliza Taylor       1.00      1.00      1.00        24
        pins_Elizabeth Lail       1.00      1.00      1.00        23
         pins_Emilia Clarke       1.00      0.97      0.98        31
            pins_Emma Stone       1.00      0.97      0.98        30
           pins_Emma Watson       0.94      1.00      0.97        32
       pins_Gwyneth Paltrow       1.00      1.00      1.00        28
           pins_Henry Cavil       1.00      1.00      1.00        29
          pins_Hugh Jackman       1.00      0.96      0.98        27
            pins_Inbar Lavi       1.00      1.00      1.00        30
           pins_Irina Shayk       0.96      1.00      0.98        23
         pins_Jake Mcdorman       1.00      1.00      1.00        24
           pins_Jason Momoa       1.00      1.00      1.00        28
     pins_Jennifer Lawrence       0.96      1.00      0.98        27
         pins_Jeremy Renner       1.00      1.00      1.00        25
        pins_Jessica Barden       1.00      0.93      0.97        30
          pins_Jimmy Fallon       1.00      1.00      1.00        30
           pins_Johnny Depp       1.00      1.00      1.00        27
           pins_Josh Radnor       1.00      1.00      1.00        30
      pins_Katharine Mcphee       1.00      1.00      1.00        27
    pins_Katherine Langford       1.00      1.00      1.00        34
          pins_Keanu Reeves       1.00      1.00      1.00        24
        pins_Krysten Ritter       1.00      1.00      1.00        26
     pins_Leonardo DiCaprio       0.94      0.97      0.96        35
         pins_Lili Reinhart       0.96      1.00      0.98        22
        pins_Lindsey Morgan       0.93      1.00      0.96        25
          pins_Lionel Messi       1.00      1.00      1.00        30
          pins_Logan Lerman       1.00      0.97      0.98        32
      pins_Madelaine Petsch       0.97      1.00      0.98        29
       pins_Maisie Williams       1.00      1.00      1.00        29
         pins_Maria Pedraza       1.00      1.00      1.00        30
    pins_Marie Avgeropoulos       1.00      1.00      1.00        24
          pins_Mark Ruffalo       1.00      1.00      1.00        27
       pins_Mark Zuckerberg       1.00      1.00      1.00        30
             pins_Megan Fox       1.00      1.00      1.00        31
           pins_Miley Cyrus       1.00      1.00      1.00        27
    pins_Millie Bobby Brown       0.97      1.00      0.98        29
       pins_Morena Baccarin       1.00      1.00      1.00        26
        pins_Morgan Freeman       1.00      1.00      1.00        30
          pins_Nadia Hilker       1.00      1.00      1.00        30
        pins_Natalie Dormer       1.00      1.00      1.00        29
       pins_Natalie Portman       1.00      1.00      1.00        25
   pins_Neil Patrick Harris       1.00      1.00      1.00        30
          pins_Pedro Alonso       1.00      1.00      1.00        30
          pins_Penn Badgley       1.00      1.00      1.00        26
            pins_Rami Malek       1.00      1.00      1.00        24
      pins_Rebecca Ferguson       1.00      1.00      1.00        27
        pins_Richard Harmon       1.00      0.97      0.98        30
               pins_Rihanna       1.00      1.00      1.00        30
        pins_Robert De Niro       1.00      1.00      1.00        23
      pins_Robert Downey Jr       1.00      1.00      1.00        35
   pins_Sarah Wayne Callies       1.00      0.96      0.98        24
          pins_Selena Gomez       1.00      0.96      0.98        28
pins_Shakira Isabel Mebarak       1.00      1.00      1.00        23
         pins_Sophie Turner       1.00      1.00      1.00        30
         pins_Stephen Amell       1.00      1.00      1.00        24
          pins_Taylor Swift       1.00      1.00      1.00        30
            pins_Tom Cruise       1.00      1.00      1.00        29
             pins_Tom Hardy       1.00      1.00      1.00        30
        pins_Tom Hiddleston       1.00      1.00      1.00        27
           pins_Tom Holland       1.00      0.96      0.98        28
    pins_Tuppence Middleton       1.00      1.00      1.00        30
        pins_Ursula Corbero       1.00      1.00      1.00        25
      pins_Wentworth Miller       1.00      1.00      1.00        27
             pins_Zac Efron       1.00      1.00      1.00        29
               pins_Zendaya       1.00      0.97      0.98        30
           pins_Zoe Saldana       1.00      0.96      0.98        28
   pins_alycia dabnem carey       1.00      1.00      1.00        32
           pins_amber heard       1.00      1.00      1.00        33
          pins_barack obama       1.00      1.00      1.00        30
        pins_barbara palvin       1.00      1.00      1.00        29
         pins_camila mendes       1.00      1.00      1.00        24
       pins_elizabeth olsen       1.00      0.94      0.97        33
            pins_ellen page       0.93      1.00      0.97        28
             pins_elon musk       1.00      1.00      1.00        30
             pins_gal gadot       0.97      0.97      0.97        30
          pins_grant gustin       1.00      1.00      1.00        27
            pins_jeff bezos       1.00      1.00      1.00        30
        pins_kiernen shipka       0.97      1.00      0.98        30
         pins_margot robbie       0.97      0.97      0.97        33
        pins_melissa fumero       1.00      1.00      1.00        23
    pins_scarlett johansson       1.00      0.97      0.98        30
             pins_tom ellis       0.97      1.00      0.99        34

                   accuracy                           0.99      2975
                  macro avg       0.99      0.99      0.99      2975
               weighted avg       0.99      0.99      0.99      2975


<Figure size 800x800 with 2 Axes>
5-fold CV: mean=0.9917 std=0.0013
Saved centroids to embeddings_cache\\centroids.npy and classes to embeddings_cache\\classes.npy
Centroid baseline accuracy: 0.9870771569745344
Suggested cosine threshold for open-set (approx TPR=0.95): 0.4617
"""

PREDICTION_REPORT_TEXT = """Loading trained model...
✅ Model loaded. Can recognize 105 classes
Found 17534 images in human_face_dataset/pins_face_recognition

Processing images...

Predicting: 100%|██████████████████████████████████████████████████████████████| 17534/17534 [1:42:42<00:00,  2.85it/s]


================================================================================
📊 PREDICTION SUMMARY
================================================================================
✅ Total images processed: 17486
✅ Correct predictions: 17442 (99.75%)
❌ Wrong predictions: 44 (0.25%)
📊 Overall Accuracy: 0.9975 (99.75%)
📊 Average confidence: 0.8239
❌ Failed to process: 48 images

================================================================================
❌ WRONG PREDICTIONS DETAILS (44 total)
================================================================================

Classes with wrong predictions:
            actual_class  wrong_count                                                        predicted_as
        pins_Brie Larson            4            pins_Emma Stone, pins_ellen page, pins_Jennifer Lawrence
     pins_Jessica Barden            3     pins_Alex Lawther, pins_kiernen shipka, pins_Danielle Panabaker
       pins_Logan Lerman            3 pins_Sarah Wayne Callies, pins_Leonardo DiCaprio, pins_Eliza Taylor
 pins_scarlett johansson            2                                   pins_gal gadot, pins_Taylor Swift
      pins_Emilia Clarke            2                           pins_Marie Avgeropoulos, pins_Irina Shayk
   pins_Brenton Thwaites            2                           pins_Leonardo DiCaprio, pins_Bobby Morley
    pins_elizabeth olsen            2                            pins_ellen page, pins_Millie Bobby Brown
            pins_Zendaya            2                               pins_Lili Reinhart, pins_Selena Gomez
        pins_Tom Holland            2                           pins_Anne Hathaway, pins_Robert Downey Jr
     pins_Tom Hiddleston            1                                                    pins_Chris Evans
       pins_Selena Gomez            1                                                    pins_Emma Watson
       pins_Taylor Swift            1                                                     pins_Emma Stone
        pins_Zoe Saldana            1                                                 pins_Lindsey Morgan
     pins_Richard Harmon            1                                              pins_Leonardo DiCaprio
        pins_amber heard            1                                                    pins_Brie Larson
          pins_gal gadot            1                                               pins_Madelaine Petsch
      pins_margot robbie            1                                                    pins_Emma Watson
pins_Sarah Wayne Callies            1                                                 pins_Lindsey Morgan
      pins_Avril Lavigne            1                                             pins_Alexandra Daddario
    pins_Natalie Portman            1                                             pins_Millie Bobby Brown
       pins_Nadia Hilker            1                                             pins_Marie Avgeropoulos
 pins_Marie Avgeropoulos            1                                                    pins_Johnny Depp
     pins_Lindsey Morgan            1                                                  pins_Anne Hathaway
  pins_Leonardo DiCaprio            1                                               pins_Brenton Thwaites
 pins_Katherine Langford            1                                               pins_Madelaine Petsch
        pins_Johnny Depp            1                                              pins_Leonardo DiCaprio
        pins_Irina Shayk            1                                                  pins_Maria Pedraza
       pins_Hugh Jackman            1                                                      pins_tom ellis
         pins_Emma Stone            1                                                  pins_margot robbie
    pins_Chris Hemsworth            1                                                    pins_Chris Evans
    pins_Morena Baccarin            1                                                  pins_camila mendes

--------------------------------------------------------------------------------
Individual wrong predictions (showing first 20):
--------------------------------------------------------------------------------
  • amber heard214_323.jpg
    Actual: pins_amber heard → Predicted: pins_Brie Larson (confidence: 0.275)
  • Avril Lavigne238_664.jpg
    Actual: pins_Avril Lavigne → Predicted: pins_Alexandra Daddario (confidence: 0.131)
  • Brenton Thwaites46_885.jpg
    Actual: pins_Brenton Thwaites → Predicted: pins_Leonardo DiCaprio (confidence: 0.273)
  • Brenton Thwaites99_936.jpg
    Actual: pins_Brenton Thwaites → Predicted: pins_Bobby Morley (confidence: 0.204)
  • Brie Larson157_994.jpg
    Actual: pins_Brie Larson → Predicted: pins_Emma Stone (confidence: 0.136)
  • Brie Larson172_1007.jpg
    Actual: pins_Brie Larson → Predicted: pins_ellen page (confidence: 0.253)
  • Brie Larson187_1021.jpg
    Actual: pins_Brie Larson → Predicted: pins_Jennifer Lawrence (confidence: 0.127)
  • Brie Larson77_1088.jpg
    Actual: pins_Brie Larson → Predicted: pins_margot robbie (confidence: 0.330)
  • Chris Hemsworth1_384.jpg
    Actual: pins_Chris Hemsworth → Predicted: pins_Chris Evans (confidence: 0.095)
  • elizabeth olsen164_1173.jpg
    Actual: pins_elizabeth olsen → Predicted: pins_ellen page (confidence: 0.355)
  • elizabeth olsen170_1179.jpg
    Actual: pins_elizabeth olsen → Predicted: pins_Millie Bobby Brown (confidence: 0.272)
  • Emilia Clarke194_952.jpg
    Actual: pins_Emilia Clarke → Predicted: pins_Marie Avgeropoulos (confidence: 0.150)
  • Emilia Clarke48_1021.jpg
    Actual: pins_Emilia Clarke → Predicted: pins_Irina Shayk (confidence: 0.069)
  • Emma Stone36_1779.jpg
    Actual: pins_Emma Stone → Predicted: pins_margot robbie (confidence: 0.282)
  • gal gadot134_1690.jpg
    Actual: pins_gal gadot → Predicted: pins_Madelaine Petsch (confidence: 0.234)
  • Hugh Jackman118_1288.jpg
    Actual: pins_Hugh Jackman → Predicted: pins_tom ellis (confidence: 0.128)
  • Irina Shayk236_2335.jpg
    Actual: pins_Irina Shayk → Predicted: pins_Maria Pedraza (confidence: 0.082)
  • Jessica Barden211_1449.jpg
    Actual: pins_Jessica Barden → Predicted: pins_Alex Lawther (confidence: 0.779)
  • Jessica Barden31_1475.jpg
    Actual: pins_Jessica Barden → Predicted: pins_kiernen shipka (confidence: 0.098)
  • Jessica Barden34_1478.jpg
    Actual: pins_Jessica Barden → Predicted: pins_Danielle Panabaker (confidence: 0.048)

  ... and 24 more wrong predictions (see CSV for details)

================================================================================
🎯 CLASSES WITH 100% ACCURACY (74 classes)
================================================================================
              class_name  total_count
       pins_Adriana Lima          213
 pins_Millie Bobby Brown          191
            pins_Rihanna          132
   pins_Rebecca Ferguson          178
         pins_Rami Malek          160
       pins_Penn Badgley          171
       pins_Pedro Alonso          125
pins_Neil Patrick Harris          116
     pins_Natalie Dormer          196
     pins_Morgan Freeman          102
        pins_Miley Cyrus          178
       pins_Keanu Reeves          158
          pins_Megan Fox          208
    pins_Mark Zuckerberg           95
       pins_Mark Ruffalo          177
       pins_Alex Lawther          152
    pins_Maisie Williams          193
   pins_Madelaine Petsch          192
       pins_Lionel Messi           86
      pins_Lili Reinhart          150
... and 54 more classes

================================================================================
⚠️ CLASSES WITH LOWEST ACCURACY (Bottom 10)
================================================================================
             class_name  correct_count  wrong_count  total_count  accuracy
      pins_Taylor Swift            129            1          130  0.992308
   pins_elizabeth olsen            219            2          221  0.990950
  pins_Brenton Thwaites            207            2          209  0.990431
     pins_Emilia Clarke            207            2          209  0.990431
pins_scarlett johansson            199            2          201  0.990050
       pins_Tom Holland            187            2          189  0.989418
      pins_Logan Lerman            209            3          212  0.985849
           pins_Zendaya            135            2          137  0.985401
    pins_Jessica Barden            138            3          141  0.978723
       pins_Brie Larson            165            4          169  0.976331

================================================================================
✅ CORRECT PREDICTIONS SAMPLE (showing 10 of 17442)
================================================================================
  ✓ Adriana Lima0_0.jpg: pins_Adriana Lima (confidence: 0.787)
  ✓ Adriana Lima101_3.jpg: pins_Adriana Lima (confidence: 0.946)
  ✓ Adriana Lima102_4.jpg: pins_Adriana Lima (confidence: 0.907)
  ✓ Adriana Lima103_5.jpg: pins_Adriana Lima (confidence: 0.752)
  ✓ Adriana Lima104_6.jpg: pins_Adriana Lima (confidence: 0.886)
  ✓ Adriana Lima105_7.jpg: pins_Adriana Lima (confidence: 0.779)
  ✓ Adriana Lima106_8.jpg: pins_Adriana Lima (confidence: 0.794)
  ✓ Adriana Lima107_9.jpg: pins_Adriana Lima (confidence: 0.930)
  ✓ Adriana Lima108_10.jpg: pins_Adriana Lima (confidence: 0.902)
  ✓ Adriana Lima109_11.jpg: pins_Adriana Lima (confidence: 0.375)

================================================================================
📁 OUTPUT FILES SAVED:
================================================================================
✅ predictions_results.csv
   → All predictions sorted (correct first, then wrong)
   → Columns: filename, actual, predicted, confidence, status, top3

✅ predictions_summary.csv
   → Per-class accuracy summary
   → Columns: class_name, correct_count, wrong_count, total_count, accuracy
================================================================================

================================================================================
❌ FAILED TO PROCESS (48 images)
================================================================================
  • Anne Hathaway203_391.jpg: No face detected
  • Avril Lavigne11_572.jpg: No face detected
  • Avril Lavigne174_619.jpg: No face detected
  • Avril Lavigne41_685.jpg: No face detected
  • barbara palvin158_800.jpg: No face detected
  • Cristiano Ronaldo209_1326.jpg: No face detected
  • Cristiano Ronaldo226_1333.jpg: No face detected
  • Eliza Taylor202_775.jpg: No face detected
  • Elizabeth Lail102_1055.jpg: No face detected
  • Elizabeth Lail102_1056.jpg: No face detected
  • Elizabeth Lail194_1117.jpg: No face detected
  • Emilia Clarke78_1050.jpg: No face detected
  • Emma Stone73_1817.jpg: No face detected
  • Hugh Jackman119_1289.jpg: No face detected
  • jeff bezos112_2049.jpg: No face detected
  • jeff bezos12_2052.jpg: No face detected
  • jeff bezos160_2068.jpg: No face detected
  • jeff bezos178_2073.jpg: No face detected
  • Jeremy Renner175_2634.jpg: No face detected
  • Johnny Depp23_1863.jpg: No face detected
  ... and 28 more

✅ Failed images list saved to: failed_predictions.csv

================================================================================
✅ PROCESSING COMPLETE!
================================================================================
"""

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Face Recognition System",
    layout="wide",
)

# -------------------------------------------------------------------
# GLOBAL CSS – white theme + round nav/predict buttons
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .block-container {
        background-color: #ffffff !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --text-main: #0f172a;
        --text-muted: #64748b;
    }
    .app-title h2 {
        color: var(--text-main);
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .stButton>button {
        border-radius: 999px !important;
        border: none !important;
        background-color: var(--primary-color) !important;
        color: #ffffff !important;
        padding: 0.30rem 0.95rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
    }
    .stButton>button:hover {
        background-color: var(--primary-hover) !important;
        color: #ffffff !important;
    }
    .nav-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.2rem;
        margin-bottom: 0.4rem;
    }
    .stAlert {
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# LOAD MODELS ONCE (cache)
# -------------------------------------------------------------------
@st.cache_resource
def init_models():
    load_models()
    return True


_ = init_models()

# -------------------------------------------------------------------
# TOP NAV ROW
# -------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

nav_row = st.container()
with nav_row:
    spacer_col, nav_col = st.columns([2, 3])
    with nav_col:
        st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
        col_home, col_train, col_pred, col_about = st.columns([1, 1.4, 1.6, 1.0])

        with col_home:
            if st.button("Home"):
                st.session_state["page"] = "Home"
        with col_train:
            if st.button("Training Report"):
                st.session_state["page"] = "Training Report"
        with col_pred:
            if st.button("Prediction Report"):
                st.session_state["page"] = "Prediction Report"
        with col_about:
            if st.button("About"):
                st.session_state["page"] = "About"

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TITLE ROW
# -------------------------------------------------------------------
title_row = st.container()
with title_row:
    st.markdown(
        """
        <div class="app-title">
            <h2>Face Recognition System</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

page = st.session_state["page"]

# -------------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------------
if page == "Home":
    st.markdown("### 🔍 Face Recognition Demo")
    st.write("Use the controls on the left to select an image from the dataset and run prediction.")

    st.sidebar.header("Image Selection")

    person_folders = sorted([f for f in DATA_ROOT.iterdir() if f.is_dir()])
    person_names = [folder.name for folder in person_folders]

    selected_person = st.sidebar.selectbox("Select Person Folder", person_names)
    selected_person_path = DATA_ROOT / selected_person

    image_files = sorted(
        [
            f
            for f in selected_person_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    image_names = [img.name for img in image_files]

    selected_image_name = st.sidebar.selectbox("Select Image", image_names)
    selected_image_path = selected_person_path / selected_image_name

    st.sidebar.image(str(selected_image_path), caption="Selected Image", use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Chosen Image")
        st.image(str(selected_image_path), use_container_width=True)

    with col2:
        st.subheader("Prediction Output")
        if st.button("Predict"):
            result = predict_image(str(selected_image_path))
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success("Prediction complete")
                st.write("##### Predicted Label")
                st.write(f"**{result['predicted_label']}**")
                st.write("##### Confidence")
                st.write(f"**{result['confidence']:.4f}**")
        else:
            st.info("Click **Predict** to run model inference.")

# -------------------------------------------------------------------
# TRAINING REPORT PAGE
# -------------------------------------------------------------------
elif page == "Training Report":
    st.markdown("### Training Analysis Report")
    st.write("Classification metrics and evaluation summary from your notebook:")
    st.code(TRAINING_REPORT_TEXT, language="text")

    st.markdown("---")
    st.subheader("Normalized Confusion Matrix")

    if CONFUSION_IMAGE_PATH.exists():
        # Center image and control size
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.image(str(CONFUSION_IMAGE_PATH), width=700)
    else:
        st.info(
            f"Confusion matrix image not found at `{CONFUSION_IMAGE_PATH}`. "
            "Save your PNG there to display it here."
        )

# -------------------------------------------------------------------
# PREDICTION REPORT PAGE
# -------------------------------------------------------------------
elif page == "Prediction Report":
    st.markdown("### Prediction Report")
    st.write("Full prediction run analysis and per-class performance:")
    st.code(PREDICTION_REPORT_TEXT, language="text")



# -------------------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------------------
elif page == "About":
    st.markdown("### About This Project")

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        st.markdown(readme_text)
    else:
        st.write(
            """
            This app demonstrates a deployable face recognition system
            built with a FaceNet backbone (InceptionResnetV1) and an SVM classifier
            trained on the `pins_face_recognition` dataset.
            """
        )

    st.markdown("---")
    st.write("Developed by **Karan**")
