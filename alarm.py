"""
Driver Fatigue AI - Prototype
alarm.py : Gestion de l'alarme sonore avec pygame

L'alarme joue en boucle tant que l'alerte est active, et s'arrête
proprement dès que l'état repasse à la normale (pas de réamorçage
hachuré à chaque frame).
"""

import os

import pygame


ALARM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "alarm.wav"
)


class AlarmPlayer:
    """
    Encapsule pygame.mixer pour jouer/arrêter un son d'alarme en boucle.
    """

    def __init__(self, sound_path=ALARM_PATH, volume=0.8):
        pygame.mixer.init()
        if not os.path.exists(sound_path):
            raise FileNotFoundError(
                f"Fichier audio introuvable : {sound_path}. "
                "Vérifie que assets/alarm.wav existe."
            )
        self.sound = pygame.mixer.Sound(sound_path)
        self.sound.set_volume(volume)
        self.is_playing = False

    def start(self):
        """Démarre l'alarme en boucle, sans relancer si déjà en cours."""
        if not self.is_playing:
            self.sound.play(loops=-1)  # -1 = boucle infinie jusqu'à stop()
            self.is_playing = True

    def stop(self):
        """Arrête l'alarme, sans erreur si elle ne jouait pas déjà."""
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
        self.stop()
        pygame.mixer.quit()
