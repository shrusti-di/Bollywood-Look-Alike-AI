# ============================================================
# BOLLYWOOD CELEBRITY LOOK-ALIKE AI
# ============================================================
#
# Main Streamlit application
#
# Pipeline:
#
# User Photo
#     ↓
# Face Detection
#     ↓
# Face Quality Check
#     ↓
# Face Embedding
#     ↓
# Compare Against Celebrity Gallery
#     ↓
# Top 3 Visual Matches
#     ↓
# Result Card
#
# ============================================================

import os
import io
import pickle
import hashlib

import numpy as np
import streamlit as st

from PIL import Image, ImageDraw, ImageFont

from utils.face_detection import (
    load_face_analyzer,
    detect_best_face,
    crop_face
)

from utils.embeddings import (
    get_face_embedding,
    normalize_embedding,
    average_embeddings
)

from utils.similarity import (
    rank_celebrities,
    similarity_to_display_score
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Bollywood Look-Alike AI"

IMAGE_FOLDER = "celebrity_images"

EMBEDDING_FILE = os.path.join(
    "models",
    "celebrity_embeddings.pkl"
)

TOP_K = 3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
# NOTE: This block is intentionally written starting at column 0.
# Streamlit's markdown renderer treats any line indented 4+ spaces
# as a code block, which would print the HTML/CSS as literal text
# instead of rendering it. Do not re-indent the lines inside this
# triple-quoted string.

st.markdown(
"""
<style>

/* ---------------------------------------------------- */
/* MAIN APP                                              */
/* ---------------------------------------------------- */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            #35164b 0%,
            #100b17 30%,
            #07070a 70%,
            #050507 100%
        );

    color: white;
}


/* ---------------------------------------------------- */
/* HERO                                                  */
/* ---------------------------------------------------- */

.hero {
    text-align: center;
    padding: 25px 10px 35px 10px;
}

.hero-title {
    font-size: 54px;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #bdb9c7;
    font-size: 19px;
    max-width: 850px;
    margin: auto;
    line-height: 1.6;
}


/* ---------------------------------------------------- */
/* SECTION TITLES                                        */
/* ---------------------------------------------------- */

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* ---------------------------------------------------- */
/* RESULT CARD                                           */
/* ---------------------------------------------------- */

.result-container {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.10),
            rgba(255,255,255,0.035)
        );

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 28px;

    padding: 30px;

    margin-top: 20px;

    box-shadow:
        0 20px 60px rgba(0,0,0,0.40);
}


.result-heading {
    text-align: center;

    font-size: 17px;

    color: #aaa5b5;

    text-transform: uppercase;

    letter-spacing: 2px;

    margin-bottom: 10px;
}


.celebrity-name {
    text-align: center;

    font-size: 38px;

    font-weight: 900;

    margin-top: 15px;
}


.score {
    text-align: center;

    font-size: 25px;

    font-weight: 700;

    margin-top: 8px;
}


/* ---------------------------------------------------- */
/* TOP 3 CARDS                                          */
/* ---------------------------------------------------- */

.match-card {
    background:
        rgba(255,255,255,0.055);

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 22px;

    padding: 18px;

    text-align: center;

    min-height: 420px;
}


.rank {
    font-size: 25px;

    font-weight: 900;

    margin-bottom: 8px;
}


.match-name {
    font-size: 21px;

    font-weight: 800;

    margin-top: 10px;
}


.match-score {
    color: #c7c2d0;

    font-size: 16px;

    margin-top: 5px;
}


/* ---------------------------------------------------- */
/* INFORMATION BOX                                      */
/* ---------------------------------------------------- */

.info-box {
    background:
        rgba(255,255,255,0.045);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 20px;

    margin-top: 20px;

    color: #c4c1ca;

    line-height: 1.7;
}


/* ---------------------------------------------------- */
/* FOOTER                                               */
/* ---------------------------------------------------- */

.footer {
    text-align: center;

    color: #d8d5e0;

    padding: 45px 0 20px 0;

    font-size: 14px;
}


.footer-credit {
    color: #ffd166;

    font-weight: 700;

    font-size: 15px;
}


/* ---------------------------------------------------- */
/* DISCLAIMER                                           */
/* ---------------------------------------------------- */

.disclaimer {
    text-align: center;

    color: #777481;

    font-size: 12px;

    line-height: 1.6;

    margin-top: 25px;

    padding: 10px 20px;
}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================
# Same fix applied here: zero-indent the HTML string.

st.markdown(
"""
<div class="hero">
<div class="hero-title">
🎬 Bollywood Look-Alike AI
</div>
<div class="hero-subtitle">
Upload a selfie and discover which Bollywood celebrity
you visually resemble most using AI-powered face
embeddings.
</div>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None


# ============================================================
# LOAD FACE ANALYZER
# ============================================================

@st.cache_resource
def initialize_face_analyzer():

    return load_face_analyzer()


with st.spinner("🤖 Loading face AI model..."):

    face_analyzer = initialize_face_analyzer()


# ============================================================
# LOAD CELEBRITY EMBEDDINGS
# ============================================================

@st.cache_resource
def load_saved_embeddings():

    if not os.path.exists(EMBEDDING_FILE):
        return None

    try:

        with open(
            EMBEDDING_FILE,
            "rb"
        ) as file:

            data = pickle.load(file)

        return data

    except Exception as error:

        st.warning(
            f"Could not load saved embeddings: {error}"
        )

        return None


celebrity_database = load_saved_embeddings()


# ============================================================
# BUILD CELEBRITY DATABASE IF NEEDED
# ============================================================

def build_celebrity_database():

    database = {}

    if not os.path.exists(IMAGE_FOLDER):

        return database


    celebrity_folders = sorted(
        [
            folder
            for folder in os.listdir(IMAGE_FOLDER)
            if os.path.isdir(
                os.path.join(
                    IMAGE_FOLDER,
                    folder
                )
            )
        ]
    )


    # ------------------------------------------------------
    # Visible progress feedback.
    #
    # Face detection + embedding on CPU is the slow part of
    # this pipeline, and it used to run behind a single static
    # spinner with no indication of progress. This shows which
    # celebrity is currently being processed and how far
    # through the gallery the build is, so a long build reads
    # as "working" rather than "frozen".
    # ------------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    total_celebrities = len(celebrity_folders)


    for folder_index, celebrity_folder in enumerate(celebrity_folders):

        status_text.text(
            f"Processing {folder_index + 1}/{total_celebrities}: "
            f"{format_celebrity_name(celebrity_folder)}"
        )

        folder_path = os.path.join(
            IMAGE_FOLDER,
            celebrity_folder
        )


        image_files = sorted(
            [
                file
                for file in os.listdir(
                    folder_path
                )
                if file.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    )
                )
            ]
        )


        reference_embeddings = []

        reference_images = []


        for image_file in image_files:

            image_path = os.path.join(
                folder_path,
                image_file
            )


            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                detected_face = detect_best_face(
                    face_analyzer,
                    image
                )

                if detected_face is None:
                    continue

                face = crop_face(
                    detected_face,
                    image
                )

                embedding = get_face_embedding(
                    detected_face
                )


                if embedding is None:
                    continue


                embedding = normalize_embedding(
                    embedding
                )


                reference_embeddings.append(
                    embedding
                )

                reference_images.append(
                    image
                )


            except Exception as error:

                print(
                    f"Skipping {image_path}: {error}"
                )


        if len(reference_embeddings) == 0:

            progress_bar.progress(
                (folder_index + 1) / total_celebrities
            )

            continue


        averaged_embedding = average_embeddings(
            reference_embeddings
        )


        database[celebrity_folder] = {

            "name": format_celebrity_name(
                celebrity_folder
            ),

            "embedding":
                averaged_embedding,

            "reference_count":
                len(reference_embeddings),

            "image":
                reference_images[0]

        }


        progress_bar.progress(
            (folder_index + 1) / total_celebrities
        )


    status_text.empty()

    progress_bar.empty()


    return database


# ============================================================
# CELEBRITY NAME FORMATTER
# ============================================================

def format_celebrity_name(folder_name):

    return folder_name.replace(
        "_",
        " "
    ).title()


# ============================================================
# CREATE DATABASE WHEN MISSING
# ============================================================

if celebrity_database is None:

    st.info(
        "🎭 Celebrity embedding database has not been "
        "created yet. Preparing it now..."
    )

    with st.spinner(
        "Building celebrity AI gallery..."
    ):

        celebrity_database = (
            build_celebrity_database()
        )


    if len(celebrity_database) > 0:

        os.makedirs(
            "models",
            exist_ok=True
        )


        try:

            with open(
                EMBEDDING_FILE,
                "wb"
            ) as file:

                pickle.dump(
                    celebrity_database,
                    file
                )

            st.success(
                "✅ Celebrity embedding database created."
            )

        except Exception as error:

            st.warning(
                f"Database created but could not be saved: "
                f"{error}"
            )


# ============================================================
# DATABASE VALIDATION
# ============================================================

if celebrity_database is None:

    st.error(
        "❌ Celebrity database could not be created."
    )

    st.info(
        "Please create the celebrity_images folder "
        "and add celebrity folders with reference photos."
    )

    st.stop()


if len(celebrity_database) < 2:

    st.error(
        "Please add at least 2 celebrities to the gallery."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎭 Celebrity Gallery")

    st.metric(
        "Celebrities",
        len(celebrity_database)
    )


    total_reference_images = sum(
        item["reference_count"]
        for item in celebrity_database.values()
    )


    st.metric(
        "Reference Photos",
        total_reference_images
    )


    st.divider()


    st.subheader("📋 How it works")

    st.write(
        """
        **1. Upload**
        
        Upload a clear selfie.
        
        **2. Detect**
        
        AI detects the most prominent face.
        
        **3. Embed**
        
        The face is converted into a numerical
        embedding.
        
        **4. Compare**
        
        The embedding is compared with multiple
        celebrity reference photos.
        
        **5. Rank**
        
        The system returns the three closest
        visual matches.
        """
    )


    st.divider()


    st.subheader("💡 Best Results")

    st.write(
        """
        • Use one visible face
        
        • Face the camera
        
        • Use good lighting
        
        • Avoid extreme filters
        
        • Avoid heavily cropped faces
        """
    )


    st.divider()


    st.caption(
        "Powered by InsightFace + Streamlit"
    )


# ============================================================
# MAIN UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📸 Upload Your Selfie'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help=(
        "Use a clear photo with one visible face."
    )
)


# ============================================================
# PROCESS UPLOAD
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()


    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    if (
        st.session_state.uploaded_hash
        != current_hash
    ):

        st.session_state.analysis_result = None

        st.session_state.uploaded_hash = (
            current_hash
        )


    try:

        user_image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")


    except Exception:

        st.error(
            "❌ Could not read this image."
        )

        st.stop()


    # ========================================================
    # FACE DETECTION
    # ========================================================

    with st.spinner(
        "🔎 Detecting face..."
    ):

        detected_face = detect_best_face(
            face_analyzer,
            user_image
        )


    if detected_face is None:

        st.error(
            "❌ No clear face detected."
        )

        st.info(
            """
            Try another photo where:

            • Your face is clearly visible

            • You are facing roughly toward the camera

            • Lighting is good

            • Only one main face is visible
            """
        )

        st.stop()


    # ========================================================
    # FACE CROP
    # ========================================================

    cropped_face = crop_face(
        detected_face,
        user_image
    )


    # ========================================================
    # DISPLAY INPUT
    # ========================================================

    image_col, face_col = st.columns(2)


    with image_col:

        st.markdown(
            "### 📷 Uploaded Photo"
        )

        st.image(
            user_image,
            use_container_width=True
        )


    with face_col:

        st.markdown(
            "### 🧑 Detected Face"
        )

        st.image(
            cropped_face,
            use_container_width=True
        )

        st.success(
            "Face detected successfully!"
        )


    st.markdown("")


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "✨ FIND MY BOLLYWOOD LOOK-ALIKE",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyze:

        with st.spinner(
            "🧠 Comparing your face with the celebrity gallery..."
        ):

            user_embedding = get_face_embedding(
                detected_face
            )


            if user_embedding is None:

                st.error(
                    "Unable to generate a face embedding."
                )

                st.stop()


            user_embedding = normalize_embedding(
                user_embedding
            )


            # ----------------------------------------------
            # PREPARE DATABASE FOR SIMILARITY
            # ----------------------------------------------

            comparison_database = {}


            for key, information in (
                celebrity_database.items()
            ):

                comparison_database[key] = {

                    "name":
                        information["name"],

                    "embedding":
                        information["embedding"],

                    "reference_count":
                        information["reference_count"],

                    "image":
                        information["image"]

                }


            # ----------------------------------------------
            # RANK
            # ----------------------------------------------

            ranked_results = rank_celebrities(
                user_embedding,
                comparison_database
            )


            ranked_results = ranked_results[
                :TOP_K
            ]


            # ----------------------------------------------
            # STORE RESULT
            # ----------------------------------------------

            st.session_state.analysis_result = {

                "user_image":
                    user_image,

                "cropped_face":
                    cropped_face,

                "results":
                    ranked_results

            }


# ============================================================
# SHOW RESULTS
# ============================================================

if st.session_state.analysis_result is not None:

    result_data = (
        st.session_state.analysis_result
    )


    results = result_data["results"]


    if len(results) == 0:

        st.error(
            "No matching celebrities were found."
        )

        st.stop()


    # ========================================================
    # BEST MATCH
    # ========================================================

    best_match = results[0]


    best_name = best_match["name"]

    best_similarity = best_match["similarity"]

    best_display_score = (
        similarity_to_display_score(
            best_similarity
        )
    )


    st.markdown("---")


    st.markdown(
        '<div class="section-title">'
        '🏆 Your Best Bollywood Match'
        '</div>',
        unsafe_allow_html=True
    )


    result_left, result_right = (
        st.columns([1, 1])
    )


    with result_left:

        st.markdown(
            '<div class="result-container">',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-heading">'
            'YOUR CLOSEST VISUAL MATCH'
            '</div>',
            unsafe_allow_html=True
        )


        st.image(
            best_match["image"],
            use_container_width=True
        )


        st.markdown(
            f'<div class="celebrity-name">'
            f'✨ {best_name}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="score">'
            f'{best_display_score:.1f}% visual similarity'
            f'</div>',
            unsafe_allow_html=True
        )


        st.progress(
            min(
                100,
                max(
                    0,
                    int(best_display_score)
                )
            )
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with result_right:

        st.markdown(
            "### 💡 Why this match?"
        )

        # NOTE: zero-indented to avoid the Markdown code-block bug.
        st.markdown(
"""
<div class="info-box">

Among the celebrities in the current gallery,
<strong>{name}</strong> produced the highest
facial-embedding similarity with the uploaded face.

The system compares numerical representations
generated from the detected face against multiple
reference images for each celebrity.

The displayed score is a normalized visual-similarity
indicator for this application. It is <strong>not</strong>
an identity probability or proof that the person is
the celebrity.

</div>
""".format(name=best_name),
unsafe_allow_html=True
        )


        st.write("")


        st.metric(
            "Reference photos used",
            best_match["reference_count"]
        )


    # ========================================================
    # TOP 3
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🥇 Your Top 3 Matches'
        '</div>',
        unsafe_allow_html=True
    )


    result_columns = st.columns(
        len(results)
    )


    medals = [
        "🥇",
        "🥈",
        "🥉"
    ]


    for index, result in enumerate(results):

        with result_columns[index]:

            display_score = (
                similarity_to_display_score(
                    result["similarity"]
                )
            )


            st.markdown(
                '<div class="match-card">',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="rank">'
                f'{medals[index]} #{index + 1}'
                f'</div>',
                unsafe_allow_html=True
            )


            st.image(
                result["image"],
                use_container_width=True
            )


            st.markdown(
                f'<div class="match-name">'
                f'{result["name"]}'
                f'</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="match-score">'
                f'{display_score:.1f}% visual similarity'
                f'</div>',
                unsafe_allow_html=True
            )


            st.progress(
                min(
                    100,
                    max(
                        0,
                        int(display_score)
                    )
                )
            )


            st.caption(
                f"{result['reference_count']} "
                f"reference photo(s)"
            )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # FULL COMPARISON
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Comparison Details'
        '</div>',
        unsafe_allow_html=True
    )


    for index, result in enumerate(results):

        display_score = (
            similarity_to_display_score(
                result["similarity"]
            )
        )


        col1, col2, col3 = st.columns(
            [1, 5, 2]
        )


        with col1:

            st.write(
                f"**#{index + 1}**"
            )


        with col2:

            st.write(
                f"**{result['name']}**"
            )


        with col3:

            st.write(
                f"{display_score:.1f}%"
            )


    # ========================================================
    # DOWNLOAD RESULT CARD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📥 Save Your Result'
        '</div>',
        unsafe_allow_html=True
    )


    def create_result_card(
        user_image,
        celebrity_image,
        celebrity_name,
        score
    ):

        width = 1200

        height = 900

        canvas = Image.new(
            "RGB",
            (width, height),
            "#09090d"
        )


        draw = ImageDraw.Draw(
            canvas
        )


        # ----------------------------------------------------
        # Fonts
        # ----------------------------------------------------

        try:

            title_font = ImageFont.truetype(
                "arial.ttf",
                58
            )

            subtitle_font = ImageFont.truetype(
                "arial.ttf",
                28
            )

            name_font = ImageFont.truetype(
                "arial.ttf",
                48
            )

            score_font = ImageFont.truetype(
                "arial.ttf",
                36
            )

        except:

            title_font = ImageFont.load_default()

            subtitle_font = ImageFont.load_default()

            name_font = ImageFont.load_default()

            score_font = ImageFont.load_default()


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        title = "YOUR BOLLYWOOD LOOK-ALIKE"


        bbox = draw.textbbox(
            (0, 0),
            title,
            font=title_font
        )


        title_width = (
            bbox[2] - bbox[0]
        )


        draw.text(
            (
                (width - title_width) / 2,
                50
            ),
            title,
            fill="white",
            font=title_font
        )


        # ----------------------------------------------------
        # User image
        # ----------------------------------------------------

        user_copy = user_image.copy()

        user_copy.thumbnail(
            (420, 420)
        )


        user_x = (
            100 +
            (420 - user_copy.width) // 2
        )

        user_y = 180


        canvas.paste(
            user_copy,
            (
                user_x,
                user_y
            )
        )


        draw.text(
            (
                210,
                620
            ),
            "YOUR PHOTO",
            fill="#bbbbbb",
            font=subtitle_font
        )


        # ----------------------------------------------------
        # Celebrity image
        # ----------------------------------------------------

        celebrity_copy = (
            celebrity_image.copy()
        )


        celebrity_copy.thumbnail(
            (420, 420)
        )


        celebrity_x = (
            680 +
            (420 - celebrity_copy.width) // 2
        )


        celebrity_y = 180


        canvas.paste(
            celebrity_copy,
            (
                celebrity_x,
                celebrity_y
            )
        )


        draw.text(
            (
                760,
                620
            ),
            "MATCH",
            fill="#bbbbbb",
            font=subtitle_font
        )


        # ----------------------------------------------------
        # Celebrity name
        # ----------------------------------------------------

        bbox = draw.textbbox(
            (0, 0),
            celebrity_name,
            font=name_font
        )


        name_width = (
            bbox[2] - bbox[0]
        )


        draw.text(
            (
                (width - name_width) / 2,
                680
            ),
            celebrity_name,
            fill="white",
            font=name_font
        )


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score_text = (
            f"{score:.1f}% visual similarity"
        )


        bbox = draw.textbbox(
            (0, 0),
            score_text,
            font=score_font
        )


        score_width = (
            bbox[2] - bbox[0]
        )


        draw.text(
            (
                (width - score_width) / 2,
                750
            ),
            score_text,
            fill="#c9c4d0",
            font=score_font
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output = io.BytesIO()

        canvas.save(
            output,
            format="PNG"
        )


        output.seek(0)

        return output


    result_card = create_result_card(
        result_data["cropped_face"],
        best_match["image"],
        best_name,
        best_display_score
    )


    st.download_button(
        label="⬇️ Download Result Card",
        data=result_card,
        file_name="bollywood_lookalike_result.png",
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About this project"
):

    st.write(
        """
        ### How the AI works

        This application uses a face-analysis model to detect
        a face and convert it into a numerical embedding.

        Each celebrity can have multiple reference photographs.
        Their embeddings are combined to create a more stable
        representation of that celebrity's reference gallery.

        The uploaded face is then compared with each celebrity
        representation using embedding similarity.

        The application returns the closest visual matches.

        ### Important

        This is a fun visual look-alike application.

        The similarity score is not an identity probability,
        authentication result, or claim that the uploaded person
        is the celebrity.
        """
    )


# ============================================================
# FOOTER
# ============================================================
# NOTE: zero-indented to avoid the Markdown code-block bug.

st.markdown(
"""
<div class="disclaimer">

⚠️ For entertainment and portfolio demonstration only.
Results represent approximate visual similarity and should
not be interpreted as identity verification.

</div>

<div class="footer">

🎬 Bollywood Look-Alike AI<br>
Built with Python • InsightFace • NumPy • Streamlit<br>
<span class="footer-credit">Developed by Shrusti Diggavi</span><br>
IPEC Solutions Private Limited, Bangalore

</div>
""",
unsafe_allow_html=True
)