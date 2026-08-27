# ============================================================
# SIMILARITY UTILITIES
# ============================================================
#
# Handles:
#   1. Cosine similarity
#   2. Celebrity ranking
#   3. Similarity score normalization
#   4. Top-N matching
#
# ============================================================

import numpy as np


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):
    """
    Calculate cosine similarity between two embeddings.

    Returns a value approximately between -1 and 1.

    For normalized face embeddings, higher values indicate
    greater similarity in the model's embedding space.
    """

    if embedding_a is None:
        return 0.0

    if embedding_b is None:
        return 0.0


    a = np.asarray(
        embedding_a,
        dtype=np.float32
    ).reshape(-1)


    b = np.asarray(
        embedding_b,
        dtype=np.float32
    ).reshape(-1)


    if a.size == 0 or b.size == 0:
        return 0.0


    if a.shape != b.shape:
        raise ValueError(
            "Embedding dimensions do not match."
        )


    norm_a = np.linalg.norm(
        a
    )

    norm_b = np.linalg.norm(
        b
    )


    if norm_a == 0 or norm_b == 0:
        return 0.0


    similarity = np.dot(
        a,
        b
    ) / (
        norm_a *
        norm_b
    )


    return float(
        np.clip(
            similarity,
            -1.0,
            1.0
        )
    )


# ============================================================
# RANK CELEBRITIES
# ============================================================

def rank_celebrities(
    user_embedding,
    celebrity_database
):
    """
    Compare the user's embedding against all celebrity
    embeddings and return them ranked from highest similarity
    to lowest similarity.

    Parameters
    ----------
    user_embedding:
        User's normalized face embedding.

    celebrity_database:
        Dictionary containing celebrity information.

    Returns
    -------
    list of dictionaries
    """

    if user_embedding is None:

        return []


    results = []


    for celebrity_id, data in (
        celebrity_database.items()
    ):

        celebrity_embedding = data.get(
            "embedding"
        )


        if celebrity_embedding is None:

            continue


        try:

            similarity = cosine_similarity(
                user_embedding,
                celebrity_embedding
            )

        except ValueError:

            continue


        results.append({

            "id":
                celebrity_id,

            "name":
                data.get(
                    "name",
                    celebrity_id
                ),

            "similarity":
                similarity,

            "reference_count":
                data.get(
                    "reference_count",
                    0
                ),

            "image":
                data.get(
                    "image"
                )

        })


    # --------------------------------------------------------
    # Highest similarity first
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )


    return results


# ============================================================
# TOP N
# ============================================================

def get_top_matches(
    user_embedding,
    celebrity_database,
    top_n=3
):
    """
    Return the top N celebrity matches.
    """

    results = rank_celebrities(
        user_embedding,
        celebrity_database
    )


    return results[
        :top_n
    ]


# ============================================================
# SIMILARITY → DISPLAY SCORE
# ============================================================

def similarity_to_display_score(
    similarity
):
    """
    Convert raw cosine similarity into a user-friendly
    visual-similarity score.

    IMPORTANT:
        This is NOT an identity probability.

    It is only a presentation score used by this application.
    """

    if similarity is None:

        return 0.0


    similarity = float(
        similarity
    )


    # --------------------------------------------------------
    # Clamp the input
    # --------------------------------------------------------

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # Map similarity to a readable score.
    #
    # This deliberately avoids displaying raw cosine values
    # directly because those are difficult for normal users
    # to understand.
    # --------------------------------------------------------

    minimum_similarity = 0.20

    maximum_similarity = 0.80


    normalized = (
        similarity -
        minimum_similarity
    ) / (
        maximum_similarity -
        minimum_similarity
    )


    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    score = (
        50.0 +
        normalized * 45.0
    )


    return float(
        np.clip(
            score,
            50.0,
            95.0
        )
    )


# ============================================================
# RAW SIMILARITY DESCRIPTION
# ============================================================

def similarity_label(
    similarity
):
    """
    Convert raw similarity into a simple qualitative label.

    This should be interpreted as a visual comparison label,
    not an identity classification.
    """

    if similarity >= 0.70:

        return "Very strong visual match"

    elif similarity >= 0.60:

        return "Strong visual match"

    elif similarity >= 0.50:

        return "Moderate visual match"

    elif similarity >= 0.40:

        return "Some visual similarity"

    else:

        return "Lower visual similarity"


# ============================================================
# COMPARE TWO EMBEDDINGS
# ============================================================

def compare_embeddings(
    embedding_a,
    embedding_b
):
    """
    Return detailed similarity information between two
    embeddings.
    """

    similarity = cosine_similarity(
        embedding_a,
        embedding_b
    )


    return {

        "raw_similarity":
            similarity,

        "display_score":
            similarity_to_display_score(
                similarity
            ),

        "label":
            similarity_label(
                similarity
            )

    }