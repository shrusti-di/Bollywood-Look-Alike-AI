# ============================================================
# FACE EMBEDDING UTILITIES
# ============================================================
#
# Handles:
#   1. Extracting face embeddings from InsightFace
#   2. Normalizing embeddings
#   3. Averaging multiple reference embeddings
#
# ============================================================


import numpy as np


# ============================================================
# GET FACE EMBEDDING
# ============================================================

def get_face_embedding(face):
    """
    Extract a face embedding from an InsightFace face object
    or a compatible embedding array.

    Parameters
    ----------
    face:
        InsightFace face object.

    Returns
    -------
    numpy.ndarray or None
    """

    if face is None:
        return None


    # --------------------------------------------------------
    # InsightFace normally provides `normed_embedding`
    # --------------------------------------------------------

    embedding = getattr(
        face,
        "normed_embedding",
        None
    )


    # --------------------------------------------------------
    # Fallback to raw embedding
    # --------------------------------------------------------

    if embedding is None:

        embedding = getattr(
            face,
            "embedding",
            None
        )


    if embedding is None:

        return None


    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    embedding = embedding.reshape(-1)


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if embedding.size == 0:

        return None


    if not np.all(
        np.isfinite(embedding)
    ):

        return None


    return embedding


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(
    embedding
):
    """
    L2-normalize an embedding.

    This makes cosine similarity equivalent to
    a simple dot product.
    """

    if embedding is None:

        return None


    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        return embedding


    return (
        embedding / norm
    ).astype(
        np.float32
    )


# ============================================================
# AVERAGE EMBEDDINGS
# ============================================================

def average_embeddings(
    embeddings
):
    """
    Average multiple normalized face embeddings.

    Example:

        Celebrity:
            photo 1 → embedding
            photo 2 → embedding
            photo 3 → embedding

        ↓

        Average embedding

    The final vector is normalized again.
    """

    if embeddings is None:

        return None


    if len(embeddings) == 0:

        return None


    valid_embeddings = []


    for embedding in embeddings:

        if embedding is None:

            continue


        normalized = normalize_embedding(
            embedding
        )


        if normalized is not None:

            valid_embeddings.append(
                normalized
            )


    if len(valid_embeddings) == 0:

        return None


    matrix = np.vstack(
        valid_embeddings
    )


    mean_embedding = np.mean(
        matrix,
        axis=0
    )


    return normalize_embedding(
        mean_embedding
    )


# ============================================================
# EMBEDDING DIMENSION
# ============================================================

def get_embedding_dimension(
    embedding
):
    """
    Return the number of dimensions in an embedding.
    """

    if embedding is None:

        return 0


    embedding = np.asarray(
        embedding
    )


    return int(
        embedding.size
    )


# ============================================================
# VALIDATE EMBEDDING
# ============================================================

def validate_embedding(
    embedding
):
    """
    Check whether an embedding is valid.
    """

    if embedding is None:

        return False


    embedding = np.asarray(
        embedding
    )


    if embedding.size == 0:

        return False


    if not np.all(
        np.isfinite(embedding)
    ):

        return False


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        return False


    return True