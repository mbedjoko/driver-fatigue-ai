"""
Driver Fatigue AI - Prototype
utils.py : Fonctions utilitaires - indices des points du visage et calculs géométriques
"""

import math

# ---------------------------------------------------------------------------
# Indices des landmarks MediaPipe Face Mesh (478 points) pour les yeux.
# Ces indices sont stables et documentés par Google, inchangés depuis
# l'ancienne API "solutions" jusqu'à la nouvelle API "Tasks".
#
# Pour l'EAR (Eye Aspect Ratio), on a besoin de 6 points par œil :
#   - 2 points horizontaux (coins gauche/droit de l'œil)
#   - 4 points verticaux (haut/bas, à 2 endroits différents)
# ---------------------------------------------------------------------------

# Oeil GAUCHE (du point de vue de la personne filmée, donc à droite de l'image
# si on n'a pas fait de flip ; comme on flip la frame dans app.py, ça correspond
# visuellement au côté gauche de l'écran)
LEFT_EYE = {
    "left_corner": 33,
    "right_corner": 133,
    "top_1": 160,
    "bottom_1": 144,
    "top_2": 158,
    "bottom_2": 153,
}

# Oeil DROIT
RIGHT_EYE = {
    "left_corner": 362,
    "right_corner": 263,
    "top_1": 385,
    "bottom_1": 380,
    "top_2": 387,
    "bottom_2": 373,
}

# Tous les indices utiles regroupés (pratique pour le dessin de debug)
LEFT_EYE_INDICES = list(LEFT_EYE.values())
RIGHT_EYE_INDICES = list(RIGHT_EYE.values())


def euclidean_distance(p1, p2):
    """Distance euclidienne entre deux points (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calcule le EAR (Eye Aspect Ratio) pour un œil donné.

    Formule standard (Soukupová & Čech, 2016) :
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Ici adaptée à nos points nommés :
        EAR = (dist(top_1, bottom_1) + dist(top_2, bottom_2))
              / (2 * dist(left_corner, right_corner))

    Plus l'œil est ouvert, plus le EAR est élevé (typiquement ~0.25-0.35).
    Quand l'œil se ferme, le EAR chute (typiquement < 0.2).

    :param landmarks: liste de points (x, y) en pixels, indexée comme les
                       478 landmarks MediaPipe.
    :param eye_indices: dict LEFT_EYE ou RIGHT_EYE (indices nommés).
    :return: valeur EAR (float), ou None si les indices sont hors limites.
    """
    try:
        left_corner = landmarks[eye_indices["left_corner"]]
        right_corner = landmarks[eye_indices["right_corner"]]
        top_1 = landmarks[eye_indices["top_1"]]
        bottom_1 = landmarks[eye_indices["bottom_1"]]
        top_2 = landmarks[eye_indices["top_2"]]
        bottom_2 = landmarks[eye_indices["bottom_2"]]
    except IndexError:
        return None

    vertical_1 = euclidean_distance(top_1, bottom_1)
    vertical_2 = euclidean_distance(top_2, bottom_2)
    horizontal = euclidean_distance(left_corner, right_corner)

    if horizontal == 0:
        return None

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def average_ear(landmarks):
    """
    Calcule le EAR moyen des deux yeux. Retourne None si un œil n'a pas
    pu être mesuré.
    """
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)

    if left is None or right is None:
        return None

    return (left + right) / 2.0
