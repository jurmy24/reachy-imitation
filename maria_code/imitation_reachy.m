% find the reachy v1 joint angles from human data
% mars 2025, MM

clear all
close all
clc

% modélisation des axes 1 à 4, bras droit

select_mode = 0; % 0 = numérique pour affichage, 1 = symbolique pour calculs


if select_mode == 1
    % valeurs symboliques
    th = sym('th',[4,1]);
    syms L1 L2 real;
    pi = sym(pi);

elseif select_mode == 0
    % valeurs numériques
    L1 = 0.19; L2 = 0.28;

    th = [0;0;0;0]; % bras vers le bas
    % th = [pi/2;0;0;0]; % bras à l'horizontale vers l'arrière
    % th = [-pi/2;0;0;0]; % bras à l'horizontale vers l'avant
    % th = [0;-pi/2;0;0]; % bras à l'horizontale vers la droite du robot
end

% paramèteres DHM des axes 1 à 4
alpha = [0;-pi/2;-pi/2;-pi/2];
d = [0;0;0;0];
r = [0; 0; -L2;0];

% matrices de passage ij
Tbase0 = mattransfo(-pi/2,0,-pi/2,L1);
T01 = Tbase0*mattransfo(alpha(1),d(1),th(1),r(1))
T12 = mattransfo(alpha(2),d(2),th(2)-pi/2,r(2));
T23 = mattransfo(alpha(3),d(3),th(3)-pi/2,r(3));
T34 = mattransfo(alpha(4),d(4),th(4),r(4));

% matrices homogènes de situation des repères Rj dans Rbase
T02 = T01*T12
T03 = T02*T23
T04 = T03*T34

if select_mode == 0

    scaleparam = 0.1;

    figure
    plot3([0],[0],[0],'r*');
    hold on
    plot3([0 T01(1,4) T02(1,4) T03(1,4) T04(1,4)],...
        [0 T01(2,4) T02(2,4) T03(2,4) T04(2,4)],...
        [0 T01(3,4) T02(3,4) T03(3,4) T04(3,4)],'k-','LineWidth',2)
    quiver3(0,0,0,1,0,0,'r--','AutoScaleFactor',scaleparam,'LineWidth',2) %xb
    quiver3(0,0,0,0,1,0,'g--','AutoScaleFactor',scaleparam,'LineWidth',2) %yb
    quiver3(0,0,0,0,0,1,'b--','AutoScaleFactor',scaleparam,'LineWidth',2) %zb

    quiver3(Tbase0(1,4),Tbase0(2,4),Tbase0(3,4),Tbase0(1,1),Tbase0(2,1),Tbase0(3,1),'r','AutoScaleFactor',scaleparam,'LineWidth',2) %x0
    quiver3(Tbase0(1,4),Tbase0(2,4),Tbase0(3,4),Tbase0(1,2),Tbase0(2,2),Tbase0(3,2),'g','AutoScaleFactor',scaleparam,'LineWidth',2) %y0
    quiver3(Tbase0(1,4),Tbase0(2,4),Tbase0(3,4),Tbase0(1,3),Tbase0(2,3),Tbase0(3,3),'b','AutoScaleFactor',scaleparam,'LineWidth',2) %z0

    % quiver3(T01(1,4),T01(2,4),T01(3,4),T01(1,1),T01(2,1),T01(3,1),'r','AutoScaleFactor',scaleparam,'LineWidth',2) %x1
    % quiver3(T01(1,4),T01(2,4),T01(3,4),T01(1,2),T01(2,2),T01(3,2),'g','AutoScaleFactor',scaleparam,'LineWidth',2) %y1
    % quiver3(T01(1,4),T01(2,4),T01(3,4),T01(1,3),T01(2,3),T01(3,3),'b','AutoScaleFactor',scaleparam,'LineWidth',2) %z1
    %
    % quiver3(T02(1,4),T02(2,4),T02(3,4),T02(1,1),T02(2,1),T02(3,1),'r','AutoScaleFactor',scaleparam,'LineWidth',2) %x2
    % quiver3(T02(1,4),T02(2,4),T02(3,4),T02(1,2),T02(2,2),T02(3,2),'g','AutoScaleFactor',scaleparam,'LineWidth',2) %y2
    % quiver3(T02(1,4),T02(2,4),T02(3,4),T02(1,3),T02(2,3),T02(3,3),'b','AutoScaleFactor',scaleparam,'LineWidth',2) %z2
    %
    % quiver3(T03(1,4),T03(2,4),T03(3,4),T03(1,1),T03(2,1),T03(3,1),'r','AutoScaleFactor',scaleparam,'LineWidth',2) %x3
    % quiver3(T03(1,4),T03(2,4),T03(3,4),T03(1,2),T03(2,2),T03(3,2),'g','AutoScaleFactor',scaleparam,'LineWidth',2) %y3
    % quiver3(T03(1,4),T03(2,4),T03(3,4),T03(1,3),T03(2,3),T03(3,3),'b','AutoScaleFactor',scaleparam,'LineWidth',2) %z3

    quiver3(T04(1,4),T04(2,4),T04(3,4),T04(1,1),T04(2,1),T04(3,1),'r','AutoScaleFactor',scaleparam,'LineWidth',2) %x4
    quiver3(T04(1,4),T04(2,4),T04(3,4),T04(1,2),T04(2,2),T04(3,2),'g','AutoScaleFactor',scaleparam,'LineWidth',2) %y4
    quiver3(T04(1,4),T04(2,4),T04(3,4),T04(1,3),T04(2,3),T04(3,3),'b','AutoScaleFactor',scaleparam,'LineWidth',2) %z4
    axis equal
    xlabel('xb'); ylabel('yb');zlabel('zb');

elseif select_mode == 1

    %------------------------------------------------------------
    % le problème d'imitation est à partir de points Oi (épaule, coude,
    % poignet), retrouver les angles du robot, avec base flottante donnée par
    % la ligne épauleG-épauleD, avec longueurs de segments inconnues a priori

    % on suppose données les coordonnées (x,y,z) dans le repère atelier/caméra ???
    % OEG = centre épaule gauche
    % OED = centre épaule droite
    % OCD = centre coude droit
    % OPD = centre poignet droit
    % et sont initialement inconnues les longueurs l1=OEG-OED, l2=OED-OCD, l3=OCD-OPD

    Obase = [0;0;0]; % en pratique, à estimer comme le milieu du segment OEG-OED

    OED = Tbase0(1:3,4); % en pratique issu du tracking
    OCD = T04(1:3,4); % en pratique issu du tracking

    OED = simplify(OED)
    OCD = simplify(OCD)

    % résultat calculs symboliques
    % OED =
    %
    %  0
    % L1
    %  0
    %
    %
    % OCD =
    %
    % -L2*cos(th2)*sin(th1)
    %      L1 + L2*sin(th2)
    % -L2*cos(th1)*cos(th2)
    % ==> inverser cette relation pour trouver th1 et th2 (quid de L1 et
    % L2 ?)

end

% à chaque pas de temps on peut connaitre les longueurs l1, l2, l3...
% probablement valeurs bruitées - faut il fixer via pose prédéfinie au
% début ?

