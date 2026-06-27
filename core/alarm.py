"""
Driver Fatigue AI - Prototype
alarm.py : Gestion de l'alarme sonore avec pygame

L'alarme joue en boucle tant que l'alerte est active, et s'arrête
proprement dès que l'état repasse à la normale (pas de réamorçage
hachuré à chaque frame).

IMPORTANT - pygame est OPTIONNEL :
Sur certains environnements (notamment Streamlit Cloud), pygame ne peut
pas s'installer (pas de wheel précompilé pour la version de Python utilisée,
et compiler depuis les sources échoue sans les libs système SDL2/freetype).
De plus, même si pygame fonctionnait sur un serveur cloud, le son jouerait
sur les haut-parleurs du SERVEUR, pas ceux de l'utilisateur - donc ça
n'aurait aucune utilité pratique en déploiement web.

Pour que l'app reste fonctionnelle partout (avec son en local, sans son
mais sans crash sur le cloud), l'import de pygame est protégé : si pygame
n'est pas disponible, AlarmPlayer bascule silencieusement sur un mode
"sans son" (no-op) au lieu de planter au démarrage.
"""

import os

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


ALARM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "alarm.wav"
)


class AlarmPlayer:
    """
    Encapsule pygame.mixer pour jouer/arrêter un son d'alarme en boucle.

    Si pygame n'est pas installé (ex: Streamlit Cloud), cette classe
    fonctionne en mode "silencieux" : toutes les méthodes restent
    utilisables sans erreur, mais aucun son n'est joué.
    """

    def __init__(self, sound_path=ALARM_PATH, volume=0.8):
        self.enabled = PYGAME_AVAILABLE
        self.is_playing = False
        self.sound = None

        if not self.enabled:
            return

        try:
            pygame.mixer.init()
            if not os.path.exists(sound_path):
                # Pas de fichier audio : on désactive plutôt que de planter,
                # l'alerte visuelle reste fonctionnelle dans tous les cas.
                self.enabled = False
                return
            self.sound = pygame.mixer.Sound(sound_path)
            self.sound.set_volume(volume)
        except Exception:
            # Tout échec d'initialisation audio (pas de device, driver
            # manquant, etc.) bascule en mode silencieux plutôt que de
            # faire planter toute l'application.
            self.enabled = False

    def start(self):
        """Démarre l'alarme en boucle, sans relancer si déjà en cours."""
        if not self.enabled:
            return
        if not self.is_playing:
            self.sound.play(loops=-1)  # -1 = boucle infinie jusqu'à stop()
            self.is_playing = True

    def stop(self):
        """Arrête l'alarme, sans erreur si elle ne jouait pas déjà."""
        if not self.enabled:
            return
        if self.is_playing:
            self.sound.stop()
            self.is_playing = False

    def set_active(self, should_play):
        """
        Pratique : appelle start() ou stop() selon une condition booléenne,
        en une seule ligne depuis la boucle principale.
        """
        if should_play:
            self.start()
        else:
            self.stop()

    def close(self):
        if not self.enabled:
            return
        self.stop()
        pygame.mixer.quit()
