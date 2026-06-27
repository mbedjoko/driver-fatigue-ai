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


# ---------------------------------------------------------------------------
# Phase 4 : détection des yeux fermés (fatigue) vs simple clignement
# ---------------------------------------------------------------------------

# Seuil EAR sous lequel on considère l'œil "fermé".
# Calibré entre une valeur "yeux ouverts" typique (~0.30+) et
# "yeux fermés" typique (~0.05-0.10), avec marge de sécurité.
EAR_CLOSED_THRESHOLD = 0.21

# Un clignement normal dure ~150-400 ms. On veut ignorer ça et ne détecter
# qu'une fermeture PROLONGÉE (signe de micro-sommeil).
EYES_CLOSED_DURATION_ALERT = 1.5  # secondes


class EyeStateTracker:
    """
    Suit l'état des yeux (ouverts/fermés) au fil du temps pour distinguer
    un clignement normal d'une fermeture prolongée signalant la fatigue.
    """

    def __init__(self, ear_threshold=EAR_CLOSED_THRESHOLD,
                 alert_duration=EYES_CLOSED_DURATION_ALERT):
        self.ear_threshold = ear_threshold
        self.alert_duration = alert_duration

        self.eyes_closed = False        # état courant (fermés ou non)
        self.closed_since = None        # timestamp (time.time()) du début de fermeture
        self.is_drowsy_alert = False    # True si fermeture > alert_duration
        self.blink_count = 0            # nombre de clignements détectés depuis le démarrage

    def update(self, ear, current_time):
        """
        Met à jour l'état avec la valeur EAR courante et l'horodatage actuel
        (en secondes, ex: time.time()).

        :param ear: valeur EAR moyenne actuelle, ou None si non mesurable.
        :param current_time: timestamp actuel en secondes.
        :return: dict avec l'état courant (voir ci-dessous).
        """
        if ear is None:
            # Pas de visage/yeux détectés : on remet à zéro par sécurité,
            # on ne veut pas déclencher une alarme sur une absence de données.
            self.eyes_closed = False
            self.closed_since = None
            self.is_drowsy_alert = False
            return self._state(ear, closed_duration=0.0)

        currently_closed = ear < self.ear_threshold

        if currently_closed and not self.eyes_closed:
            # Transition ouvert -> fermé : on démarre le chrono
            self.eyes_closed = True
            self.closed_since = current_time
            self.is_drowsy_alert = False

        elif currently_closed and self.eyes_closed:
            # Toujours fermés : on vérifie la durée
            closed_duration = current_time - self.closed_since
            if closed_duration >= self.alert_duration:
                self.is_drowsy_alert = True

        elif not currently_closed and self.eyes_closed:
            # Transition fermé -> ouvert : c'était un clignement (ou une alerte qui se termine)
            closed_duration = current_time - self.closed_since
            if closed_duration < self.alert_duration:
                self.blink_count += 1
            self.eyes_closed = False
            self.closed_since = None
            self.is_drowsy_alert = False

        closed_duration = (current_time - self.closed_since) if self.closed_since else 0.0
        return self._state(ear, closed_duration)

    def _state(self, ear, closed_duration):
        return {
            "ear": ear,
            "eyes_closed": self.eyes_closed,
            "closed_duration": closed_duration,
            "is_drowsy_alert": self.is_drowsy_alert,
            "blink_count": self.blink_count,
        }


# ---------------------------------------------------------------------------
# Phase 5 : détection des bâillements (MAR - Mouth Aspect Ratio)
# ---------------------------------------------------------------------------

# Indices des landmarks MediaPipe Face Mesh pour la bouche.
# Stables et documentés, identiques entre ancienne et nouvelle API.
MOUTH = {
    "left_corner": 61,
    "right_corner": 291,
    "top_outer_1": 81,
    "bottom_outer_1": 178,
    "top_inner": 13,
    "bottom_inner": 14,
    "top_outer_2": 311,
    "bottom_outer_2": 402,
}

MOUTH_INDICES = list(MOUTH.values())

# Seuil MAR au-delà duquel on considère la bouche "grande ouverte".
# Un MAR élevé soutenu (pas juste un pic bref comme en parlant) = bâillement.
MAR_YAWN_THRESHOLD = 0.55

# Durée minimale (secondes) d'ouverture soutenue pour valider un bâillement.
# Un vrai bâillement dure généralement 3-7 secondes ; on met un seuil bas
# (1.0s) pour détecter tôt, tout en filtrant les ouvertures très brèves
# (rire, parole) qui ne durent pas.
YAWN_DURATION_THRESHOLD = 1.0


def mouth_aspect_ratio(landmarks):
    """
    Calcule le MAR (Mouth Aspect Ratio) à partir des landmarks de la bouche.

    Formule (analogue au EAR) :
        MAR = (dist verticale moyenne) / (dist horizontale)

    :param landmarks: liste de points (x, y) en pixels (478 landmarks).
    :return: valeur MAR (float), ou None si indices hors limites.
    """
    try:
        left_corner = landmarks[MOUTH["left_corner"]]
        right_corner = landmarks[MOUTH["right_corner"]]
        top_outer_1 = landmarks[MOUTH["top_outer_1"]]
        bottom_outer_1 = landmarks[MOUTH["bottom_outer_1"]]
        top_inner = landmarks[MOUTH["top_inner"]]
        bottom_inner = landmarks[MOUTH["bottom_inner"]]
        top_outer_2 = landmarks[MOUTH["top_outer_2"]]
        bottom_outer_2 = landmarks[MOUTH["bottom_outer_2"]]
    except IndexError:
        return None

    vertical_1 = euclidean_distance(top_outer_1, bottom_outer_1)
    vertical_2 = euclidean_distance(top_inner, bottom_inner)
    vertical_3 = euclidean_distance(top_outer_2, bottom_outer_2)
    horizontal = euclidean_distance(left_corner, right_corner)

    if horizontal == 0:
        return None

    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
    return mar


class YawnTracker:
    """
    Suit l'ouverture de la bouche au fil du temps pour distinguer un
    bâillement (ouverture large et soutenue) d'une ouverture brève
    (parole, rire, expression rapide).
    """

    def __init__(self, mar_threshold=MAR_YAWN_THRESHOLD,
                 duration_threshold=YAWN_DURATION_THRESHOLD):
        self.mar_threshold = mar_threshold
        self.duration_threshold = duration_threshold

        self.mouth_open = False
        self.open_since = None
        self.is_yawning = False     # True dès que la durée soutenue est atteinte
        self.yawn_count = 0         # nombre de bâillements validés depuis le démarrage
        self._counted_current = False  # évite de compter 2x le même bâillement

    def update(self, mar, current_time):
        """
        Met à jour l'état avec la valeur MAR courante et l'horodatage actuel.

        :param mar: valeur MAR actuelle, ou None si non mesurable.
        :param current_time: timestamp actuel en secondes.
        :return: dict avec l'état courant.
        """
        if mar is None:
            self.mouth_open = False
            self.open_since = None
            self.is_yawning = False
            self._counted_current = False
            return self._state(mar, open_duration=0.0)

        currently_open = mar > self.mar_threshold

        if currently_open and not self.mouth_open:
            self.mouth_open = True
            self.open_since = current_time
            self.is_yawning = False
            self._counted_current = False

        elif currently_open and self.mouth_open:
            open_duration = current_time - self.open_since
            if open_duration >= self.duration_threshold:
                self.is_yawning = True
                if not self._counted_current:
                    self.yawn_count += 1
                    self._counted_current = True

        elif not currently_open and self.mouth_open:
            self.mouth_open = False
            self.open_since = None
            self.is_yawning = False
            self._counted_current = False

        open_duration = (current_time - self.open_since) if self.open_since else 0.0
        return self._state(mar, open_duration)

    def _state(self, mar, open_duration):
        return {
            "mar": mar,
            "mouth_open": self.mouth_open,
            "open_duration": open_duration,
            "is_yawning": self.is_yawning,
            "yawn_count": self.yawn_count,
        }
