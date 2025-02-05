https://docs.pollen-robotics.com/sdk/first-moves/kinematics/

Projet Imitation
Pole Projet Rbotique P19
CentraleSupélec

Contributeurs :
Cheikh Rouhou Mohamed
Maes Alexis
Lafaury Marin
Champaney Matéo

Encadrants :
Makarov Maria
Tao Xavier

Prise en main du projet:
Pour pouvoir utiliser le robot et la vision par caméra de profondeur, nous avons mis en place un environnement virtuel python et installé diffèrentes librairies. Vous
pouvez copier les lignes suivantes et les executer dans un terminal Bash :
python -m venv env

source env/Scripts/activate

pip3 install reachy-sdk
git clone https://github.com/pollen-robotics/reachy-sdk
pip3 install -e reachy-sdk

pip install numpy
pip install time
pip install math
pip install mediapipe
pip install cv2
pip install scipy
pip install pyrealsense2
pip install random
pip install os

Organisation :
Nous avons développer différents programme. Certains servent uniquement à se connecter au robot, d'autres servent uniquement à des expèriences de visions, d'autres encore combinent vision et controle
