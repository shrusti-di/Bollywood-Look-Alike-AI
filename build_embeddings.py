# ============================================================
# BUILD CELEBRITY EMBEDDINGS
# ============================================================
#
# Run this file BEFORE starting Streamlit:
#
#     python build_embeddings.py
#
# It will:
#
#   1. Read celebrity images
#   2. Detect the best face
#   3. Crop the face
#   4. Generate face embeddings
#   5. Average multiple reference embeddings
#   6. Save the database to:
#
#       models/celebrity_embeddings.pkl
#
# ------------------------------------------------------------
# PERFORMANCE NOTE
# ------------------------------------------------------------
#
# Face detection + embedding on CPU is the slow part of this
# pipeline. With many celebrities and many reference photos
# each, running everything through one CPU core one image at
# a time gets slow fast (e.g. 54 celebrities x 30 photos =
# 1,620 images processed serially).
#
# This version processes celebrities in PARALLEL worker
# processes (one InsightFace instance per process, reused for
# all of that worker's images), so the work is spread across
# every CPU core available on the machine running this script.
#
# Tips for best results:
#
#   - Run this script on your own development machine (which
#     usually has several CPU cores) rather than inside a
#     constrained hosting environment (many free-tier hosts
#     only give you 1 CPU core, where parallelism won't help).
#     Then just ship/upload the resulting
#     models/celebrity_embeddings.pkl file alongside your app.
#     The deployed Streamlit app only needs to LOAD that file,
#     not rebuild it.
#
#   - Consider whether you need 30 reference photos per
#     celebrity. In practice, 8-12 varied, clear photos give a
#     very similar averaged embedding to 30, at a fraction of
#     the processing time.
#
# ============================================================

import os
import pickle
import multiprocessing as mp

from PIL import Image

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


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_FOLDER = "celebrity_images"

MODEL_FOLDER = "models"

OUTPUT_FILE = os.path.join(
    MODEL_FOLDER,
    "celebrity_embeddings.pkl"
)


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# Set this to a specific number to override how many worker
# processes are used (e.g. NUM_WORKERS = 4). Leave as None to
# auto-detect based on available CPU cores.
NUM_WORKERS = None


# ============================================================
# CELEBRITY NAME
# ============================================================

def format_celebrity_name(
    folder_name
):
    """
    Convert folder names such as:

        deepika_padukone

    into:

        Deepika Padukone
    """

    return folder_name.replace(
        "_",
        " "
    ).title()


# ============================================================
# PER-WORKER FACE ANALYZER
# ============================================================
#
# Each worker process loads its OWN InsightFace instance once,
# the first time it's needed, and reuses it for every celebrity
# folder that process is assigned. This avoids reloading the
# model for every single image, while still letting multiple
# processes run at the same time.
# ============================================================

_worker_face_analyzer = None


def _init_worker():

    global _worker_face_analyzer

    _worker_face_analyzer = load_face_analyzer()


# ============================================================
# PROCESS A SINGLE CELEBRITY FOLDER (RUNS INSIDE A WORKER)
# ============================================================

def process_celebrity_folder(task):

    celebrity_folder, folder_path = task

    global _worker_face_analyzer

    face_analyzer = _worker_face_analyzer


    celebrity_name = format_celebrity_name(
        celebrity_folder
    )


    log_lines = []


    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = sorted(
        [
            file
            for file in os.listdir(folder_path)
            if file.lower().endswith(
                SUPPORTED_EXTENSIONS
            )
        ]
    )


    if len(image_files) == 0:

        log_lines.append(
            f"⚠️  {celebrity_name}: no images found"
        )

        return celebrity_folder, None, "\n".join(log_lines)


    log_lines.append(
        f"🎬 {celebrity_name} "
        f"({len(image_files)} reference images)"
    )


    reference_embeddings = []

    reference_images = []


    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

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

                log_lines.append(
                    f"   ❌ {image_file}: no face"
                )

                continue


            embedding = get_face_embedding(
                detected_face
            )


            if embedding is None:

                log_lines.append(
                    f"   ❌ {image_file}: embedding failed"
                )

                continue


            embedding = normalize_embedding(
                embedding
            )


            if embedding is None:

                log_lines.append(
                    f"   ❌ {image_file}: invalid embedding"
                )

                continue


            reference_embeddings.append(
                embedding
            )

            reference_images.append(
                image
            )


        except Exception as error:

            log_lines.append(
                f"   ❌ {image_file}: error: {error}"
            )


    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if len(reference_embeddings) == 0:

        log_lines.append(
            f"   ❌ No valid face embeddings for "
            f"{celebrity_name}"
        )

        return celebrity_folder, None, "\n".join(log_lines)


    averaged_embedding = average_embeddings(
        reference_embeddings
    )


    if averaged_embedding is None:

        log_lines.append(
            "   ❌ Could not create average embedding."
        )

        return celebrity_folder, None, "\n".join(log_lines)


    entry = {

        "name":
            celebrity_name,

        "embedding":
            averaged_embedding,

        "reference_count":
            len(reference_embeddings),

        "image":
            reference_images[0]

    }


    log_lines.append(
        f"   ✅ {len(reference_embeddings)}/"
        f"{len(image_files)} valid"
    )


    return celebrity_folder, entry, "\n".join(log_lines)


# ============================================================
# BUILD DATABASE (PARALLEL ACROSS CELEBRITIES)
# ============================================================

def build_database():

    database = {}


    # --------------------------------------------------------
    # Check celebrity folder
    # --------------------------------------------------------

    if not os.path.exists(IMAGE_FOLDER):

        print(
            f"\nERROR: '{IMAGE_FOLDER}' folder "
            f"does not exist."
        )

        return database


    # --------------------------------------------------------
    # Get celebrity folders
    # --------------------------------------------------------

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


    if len(celebrity_folders) == 0:

        print(
            "\nERROR: No celebrity folders found."
        )

        print(
            "Example:"
        )

        print(
            "celebrity_images/"
        )

        print(
            "    deepika_padukone/"
        )

        return database


    # --------------------------------------------------------
    # Decide worker count
    # --------------------------------------------------------

    available_cores = os.cpu_count() or 1

    num_workers = NUM_WORKERS or max(
        1,
        min(
            available_cores,
            len(celebrity_folders)
        )
    )


    print(
        "\n=========================================="
    )

    print(
        "   BUILDING CELEBRITY EMBEDDINGS"
    )

    print(
        "=========================================="
    )

    print(
        f"\nCelebrities found: "
        f"{len(celebrity_folders)}"
    )

    print(
        f"CPU cores available: {available_cores}"
    )

    print(
        f"Worker processes used: {num_workers}"
    )

    if num_workers == 1:

        print(
            "\n⚠️  Running with a single worker "
            "(1 CPU core detected, or NUM_WORKERS=1). "
            "Parallelism will not speed this up on this "
            "machine. If you're on a hosted/CI environment "
            "with limited CPU, consider running this script "
            "on a local machine with more cores instead, "
            "then uploading the resulting "
            f"{OUTPUT_FILE} file."
        )


    # --------------------------------------------------------
    # Build task list: one task per celebrity folder
    # --------------------------------------------------------

    tasks = [
        (
            celebrity_folder,
            os.path.join(
                IMAGE_FOLDER,
                celebrity_folder
            )
        )
        for celebrity_folder in celebrity_folders
    ]


    # --------------------------------------------------------
    # Run in parallel
    # --------------------------------------------------------

    completed = 0

    total = len(tasks)


    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker
    ) as pool:

        for celebrity_folder, entry, log in pool.imap_unordered(
            process_celebrity_folder,
            tasks
        ):

            completed += 1

            print(
                f"\n[{completed}/{total}] {log}"
            )

            if entry is not None:

                database[celebrity_folder] = entry


    return database


# ============================================================
# SAVE DATABASE
# ============================================================

def save_database(
    database
):

    if len(database) == 0:

        print(
            "\n❌ Nothing to save."
        )

        return False


    # --------------------------------------------------------
    # Create models directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save pickle
    # --------------------------------------------------------

    try:

        with open(
            OUTPUT_FILE,
            "wb"
        ) as file:

            pickle.dump(
                database,
                file
            )


        print(
            "\n=========================================="
        )

        print(
            "          EMBEDDINGS COMPLETE"
        )

        print(
            "=========================================="
        )


        print(
            f"\n✅ Celebrities processed: "
            f"{len(database)}"
        )


        total_images = sum(
            item["reference_count"]
            for item in database.values()
        )


        print(
            f"✅ Valid reference images: "
            f"{total_images}"
        )


        print(
            f"✅ Saved to:"
        )


        print(
            f"   {OUTPUT_FILE}"
        )


        return True


    except Exception as error:

        print(
            f"\n❌ Could not save database:"
            f"\n{error}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
    )

    print(
        "🎬 Bollywood Look-Alike AI"
    )

    print(
        "Celebrity Embedding Builder (parallel)"
    )


    # --------------------------------------------------------
    # Quick import check (fails fast, before spawning workers,
    # if InsightFace isn't installed at all).
    # --------------------------------------------------------

    try:

        import insightface  # noqa: F401

    except ImportError as error:

        print(
            "\n❌ Could not import InsightFace."
        )

        print(
            f"Error: {error}"
        )

        print(
            "Run: pip install insightface onnxruntime"
        )

        return


    # --------------------------------------------------------
    # Build (each worker loads its own model instance)
    # --------------------------------------------------------

    database = build_database()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_database(
        database
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # multiprocessing on some platforms (notably Windows and
    # macOS with the "spawn" start method) needs the entry
    # point guarded like this, or worker processes will
    # re-import and re-run this script recursively.

    main()