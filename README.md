# Projet Imitation, Pole Projet Robotique P19, CentraleSupélec

Date : Semestre 8 (2025)

Contributeurs (Semestre 7) :

- Cheikh, Rouhou Mohamed
- Maes, Alexis
- Lafaury, Marin
- Champaney, Matéo

Contributeurs (Semestre 8) :

- Lafaury, Marin
- Oldensand, Victor
- Gandhi, Hugo
- Ye, Yi

Encadrants :

- Makarov, Maria
- Tao, Xavier

## Documentation

https://docs.pollen-robotics.com/sdk/first-moves/kinematics/

### Prise en main du projet

Pour pouvoir utiliser le robot et la vision par caméra de profondeur, nous avons mis en place un environnement virtuel en utilisant [uv](https://docs.astral.sh/uv/). Vous
pouvez copier les lignes suivantes et les executer pour préparer l'environnement virtuel vous-même.

```bash
uv sync
```

## Le Dashboard

Quand le robot est connecté il est possible to visualiser l'état des joints et moteurs sur le site `<robot-ip>:3972` dans un navigateur.

Organisation :
Nous avons développé différents programmes. Certains servent uniquement à se connecter au robot, d'autres servent uniquement à des expèriences de visions, d'autres encore combinent vision et controle.
