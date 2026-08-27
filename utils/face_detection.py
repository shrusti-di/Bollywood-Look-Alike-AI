# ============================================================
# FACE DETECTION UTILITIES
# ============================================================
#
# Handles:
#   1. InsightFace model initialization
#   2. Face detection
#   3. Best-face selection
#   4. Face cropping
#   5. Basic face-quality validation
#
# ============================================================

import numpy as np
from PIL import Image


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def load_face_analyzer():
    """
    Load the InsightFace face-analysis model.

    Returns
    -------
    app : FaceAnalysis
        Initialized InsightFace analyzer.
    """

    try:

        from insightface.app import FaceAnalysis

    except ImportError:

        raise ImportError(
            "InsightFace is not installed. "
            "Run: pip install insightface onnxruntime"
        )


    # --------------------------------------------------------
    # Create analyzer
    #
    # Model pack comparison (from InsightFace's model zoo):
    #
    #   buffalo_l  - RetinaFace-10GF detector, ResNet50
    #                recognition, + landmarks/age/gender.
    #                326MB. Most accurate, slowest on CPU.
    #
    #   buffalo_s  - RetinaFace-500MF detector (much lighter
    #                than buffalo_l's), MobileFaceNet
    #                recognition, + landmarks/age/gender.
    #                159MB. Noticeably faster on CPU with a
    #                small accuracy trade-off. Good default
    #                for a look-alike app that just needs one
    #                clear face per photo.
    #
    #   buffalo_sc - Same lightweight detector + recognition
    #                as buffalo_s, but drops the landmark and
    #                age/gender models entirely. 16MB. Fastest
    #                option; fine here since this app doesn't
    #                use landmarks or age/gender.
    #
    # Switch back to "buffalo_l" if match quality matters more
    # than speed.
    # --------------------------------------------------------

    app = FaceAnalysis(
        name="buffalo_s",
        providers=[
            "CPUExecutionProvider"
        ]
    )


    # --------------------------------------------------------
    # Prepare model
    #
    # det_size controls the resolution used for face
    # detection. 640x640 is InsightFace's default and is
    # tuned for finding many small faces in crowded photos.
    # This app only needs to find one clear face in a selfie
    # or portrait, so 320x320 is plenty and is significantly
    # faster on CPU with no real accuracy loss for this use
    # case.
    # --------------------------------------------------------

    app.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )


    return app


# ============================================================
# PIL → NUMPY
# ============================================================

def pil_to_numpy(image):
    """
    Convert PIL image to RGB NumPy array.
    """

    if not isinstance(image, Image.Image):

        raise TypeError(
            "Expected a PIL.Image.Image object."
        )


    image = image.convert("RGB")

    return np.array(image)


# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(
    face_analyzer,
    image
):
    """
    Detect all faces in an image.

    Parameters
    ----------
    face_analyzer:
        Initialized InsightFace analyzer.

    image:
        PIL image.

    Returns
    -------
    faces:
        List of InsightFace face objects.
    """

    image_array = pil_to_numpy(
        image
    )


    faces = face_analyzer.get(
        image_array
    )


    return faces


# ============================================================
# GET FACE AREA
# ============================================================

def get_face_area(face):
    """
    Calculate the area of a detected face.
    """

    bbox = face.bbox

    x1, y1, x2, y2 = bbox

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    return float(
        width * height
    )


# ============================================================
# SELECT BEST FACE
# ============================================================

def detect_best_face(
    face_analyzer,
    image
):
    """
    Detect faces and select the most suitable face.

    Strategy:
        - Detect all faces.
        - Prefer the largest face.
        - Return None when no face is detected.

    This is useful because a user may accidentally upload
    a group photo.
    """

    faces = detect_faces(
        face_analyzer,
        image
    )


    if faces is None or len(faces) == 0:

        return None


    # --------------------------------------------------------
    # Select largest detected face
    # --------------------------------------------------------

    best_face = max(
        faces,
        key=get_face_area
    )


    return best_face


# ============================================================
# GET BOUNDING BOX
# ============================================================

def get_bounding_box(
    face,
    image_width,
    image_height,
    padding_ratio=0.30
):
    """
    Get a padded bounding box around a detected face.

    Parameters
    ----------
    face:
        InsightFace face object.

    image_width:
        Width of original image.

    image_height:
        Height of original image.

    padding_ratio:
        Extra area around the detected face.

    Returns
    -------
    x1, y1, x2, y2
    """

    x1, y1, x2, y2 = face.bbox


    # Convert to integers

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)


    face_width = max(
        1,
        x2 - x1
    )

    face_height = max(
        1,
        y2 - y1
    )


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padding_x = int(
        face_width * padding_ratio
    )

    padding_y = int(
        face_height * padding_ratio
    )


    x1 = max(
        0,
        x1 - padding_x
    )

    y1 = max(
        0,
        y1 - padding_y
    )

    x2 = min(
        image_width,
        x2 + padding_x
    )

    y2 = min(
        image_height,
        y2 + padding_y
    )


    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CROP FACE
# ============================================================

def crop_face(
    face,
    image=None
):
    """
    Crop the detected face from an image.

    Parameters
    ----------
    face:
        InsightFace face object.

    image:
        Original PIL image.

    Returns
    -------
    PIL.Image
        Cropped face image.
    """

    if image is None:

        raise ValueError(
            "Original image is required."
        )


    image = image.convert(
        "RGB"
    )


    image_width, image_height = (
        image.size
    )


    x1, y1, x2, y2 = get_bounding_box(
        face,
        image_width,
        image_height
    )


    cropped = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    return cropped


# ============================================================
# FACE QUALITY
# ============================================================

def calculate_face_quality(
    face,
    image
):
    """
    Calculate a basic face-quality score.

    The score considers:

        - Face size relative to image
        - Detection confidence

    Returns
    -------
    dict
    """

    image_width, image_height = (
        image.size
    )


    image_area = (
        image_width *
        image_height
    )


    face_area = get_face_area(
        face
    )


    if image_area <= 0:

        return {
            "score": 0.0,
            "face_ratio": 0.0,
            "confidence": 0.0
        }


    face_ratio = (
        face_area /
        image_area
    )


    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    confidence = float(
        getattr(
            face,
            "det_score",
            0.0
        )
    )


    # --------------------------------------------------------
    # Face-size score
    #
    # A face occupying around 10%+ of the image is generally
    # large enough for this application.
    # --------------------------------------------------------

    size_score = min(
        1.0,
        face_ratio / 0.10
    )


    # --------------------------------------------------------
    # Combined quality
    # --------------------------------------------------------

    quality_score = (
        0.60 * size_score +
        0.40 * confidence
    )


    quality_score = float(
        np.clip(
            quality_score,
            0.0,
            1.0
        )
    )


    return {
        "score": quality_score,
        "face_ratio": float(face_ratio),
        "confidence": confidence
    }


# ============================================================
# FACE VALIDATION
# ============================================================

def validate_face(
    face,
    image,
    minimum_quality=0.25
):
    """
    Validate whether the detected face is suitable.

    Returns
    -------
    dict containing:

        valid
        score
        face_ratio
        confidence
        message
    """

    quality = calculate_face_quality(
        face,
        image
    )


    if quality["score"] < minimum_quality:

        return {
            **quality,
            "valid": False,
            "message": (
                "The detected face is too small or "
                "unclear. Please upload a clearer photo."
            )
        }


    return {
        **quality,
        "valid": True,
        "message": (
            "Face quality is suitable for comparison."
        )
    }


# ============================================================
# FACE INFORMATION
# ============================================================

def get_face_info(
    face,
    image
):
    """
    Return useful information about the detected face.
    """

    quality = calculate_face_quality(
        face,
        image
    )


    bbox = face.bbox


    x1, y1, x2, y2 = bbox


    return {
        "bounding_box": (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ),

        "area": get_face_area(
            face
        ),

        "detection_confidence":
            float(
                getattr(
                    face,
                    "det_score",
                    0.0
                )
            ),

        "quality_score":
            quality["score"],

        "face_ratio":
            quality["face_ratio"]
    }