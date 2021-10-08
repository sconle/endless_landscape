#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2020 Reso-nance Numérique <laurent@reso-nance.org>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
# usage: python3 playRandom_RAM.py monfichier.mp4

import os, sys, cv2, random, numpy, pygame
from datetime import datetime, timedelta

# !! Ce programme est optimisé pour fonctionner sous un environnement Linux (et raspberry pi), l'utilisation sous windows
#(qui plus est sur un ordinateur "classique") n'est pas (encore) garantit sans mauvaises surprises !!

# 1 proba fixe + 1 liste
# temps min et temps max
screenSize = [1920,1080] # taille de l'écran d'affichage (colonne, ligne)
displaySize = [600,600] # taille de la fenêtre d'affichage (colonne, ligne)
videoSize = [1920,1080] # dimension de la vidéo de base (colonne, ligne)
changePlayDirectionProbas = [0.02, 0.032, 0.1, 0.01] # liste de probabilités de changer de direction de lecture
fixedPlayDirectionChangeProba = .01 # probabilité de changement de direction supplémentaire (celle ci est fixe, ce n'est pas une liste)
changeProbasEvery = [4.0, 7.0] # temps en secondes (min et max, peu importe l'ordre) au bout duquel une nouvelle proba est tirée au sort dans changePlayDirectionProbas
changeConfigEvery = [3.0,5.0] # temps en seconde (min et max, peu importe l'ordre) au bout duquel une nouvelle config est choisie
enterLoopProbas = 1/1200 # probabilités à chaque frame de rentrer dans une boucle
timeBetweenLoops = 5.0 # temps minimum où le programme doit attendre à la sortie d'une boucle avant de retester à chaque frame si l'on doit rentrer dans une nouvelle boucle (en s)
loopRangeProbaTimer = [[[2, 4], 0.5, [3.0, 4.0], [10,20]],
                       [[4, 6],0.5,[3.0, 7.0], [10,30]]] # différentes amplitudes (en frames) de boucle ainsi que les probabilités (la somme des probas doit faire 1),
                                           # d'une plage de durée (en s) ainsi que d'un nombre de frames dont la boucle se décalera
maxFPS = 20 # nombre maximal d'images par secondes souhaitées

listPointsZoom = [[[[960,540],None],[1.2,None]],
                  [[[1670,250],None],[1.2,None]],
                  [[[1733,893],None],[1.6,None]],
                  [[[230,230],None],[1.3,None]],
                  [[[1763,157],None],[1.9,None]]] # liste des coordonnées/zooms (colonne, ligne) des différents points d'interets. La liste se présente comme suit:
                                                  # [[[[X,Y],None],[Zoom,None]],...] None en 2eme position pour les coordonnées si le point atteint doit être fixe
                                                  # (sinon on tirera aléatoirement un point entre les deux coordonnées), de même pour le zoom.
                                                  
                                                  # on utilise listPointsZoom dans le code comme suit: listPointsZoom[n°config][0:coordonnées 1:zoom][0:1ere valeur 1:soit None soit une valeur qui donne un intervalle]
avgSpeed = 150 # vitesse moyenne de déplacement pendant les transitions en pixels par seconde
probaRandConfig = 1 # 0 que des configs provenant de la liste / 1 que de l'alea
randZoomInter = [1.2, 1.7] # intervalle dans lequel on ppeut piocher un zoom aléatoire
forkZoom = 0.01 # fourchette à partir de laquelle le zoom est considéré comme atteint, par exemple pour une fourchette de 0.1,
                #si l'on souhaite atteindre un zoom de 1.3 alors on considerera comme acceptable un zoom de 1.2 ou 1.4
portrait = False # portrait = False si on affiche en paysage, True si on affiche en portrait
mirror = False # inverser la video (ce n'est pas un vrai effet miroir, on tourne juste la vidéo de 180°)



# on vérifie si la somme des probas dans loopRangeProbaTimer est égale à 1
sum = 0
for rangeProbaTimer in loopRangeProbaTimer:
    sum += rangeProbaTimer[1]
if float(sum) != 1.0:
    raise Exception(f"sum of probabilities in loopRangeProba is not equal to 1.0, equal to {sum} instead")

angle = 0
if portrait:
    displaySize[0], displaySize[1] = displaySize[1], displaySize[0]
    videoSize[0], videoSize[1] = videoSize[1], videoSize[0]
    angle = 90
if mirror:
    angle += 180

# changement de coordonnée des points d'intérêts en fonction de la rotation de l'image
for config in listPointsZoom:
    if angle == 90: 
       config[0][0] = [config[0][0][1],videoSize[1]-config[0][0][0]]
    elif angle == 180:
        config[0][0] = [videoSize[0] - config[0][0][0],videoSize[1] - config[0][0][1]]
    elif angle == 270:
        config[0][0] = [videoSize[0] - config[0][0][1],config[0][0][0]]
    if config[0][1] is not None:
        if angle == 90:
           config[0][1] = [config[0][1][1],videoSize[1]-config[0][1][0]]
        elif angle == 180:
            config[0][1] = [videoSize[0] - config[0][1][0],videoSize[1] - config[0][1][1]]
        elif angle == 270:
            config[0][1] = [videoSize[0] - config[0][1][1],config[0][1][0]]

pygame.init()
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_FBDEV"] = "/dev/fb0"
os.environ['SDL_NOMOUSE'] = '1'
os.environ['SDL_VIDEO_WINDOW_POS'] = "%i,%i" % (550,550) 
os.environ['SDL_VIDEO_CENTERDED'] = '1'
print("running in headless mode using framebuffer",os.environ["SDL_FBDEV"])
pygame.mixer.init()
pygame.mouse.set_visible(False)

class ImageSequence():

    def __init__(self, videoFilename):
        self.frameCount = 0
        self.frames=[] # sauvegarde cv2.imread() pour chaque image
        self.extractFrames(videoFilename)
        self.currentFrameIndex = 0 # sur quelle image est la tête de lecture
        self.playingForward = True
        self.changePlayDirectionProba = random.choice(changePlayDirectionProbas)
        self.zoom = listPointsZoom[0][1][0]
        self.points = listPointsZoom[0][0][0]
        self.config = 0

    def extractFrames(self, videoPath):
        """ extracts frames from the video file to RAM, including cv2 -> pygame conversion in headless mode"""
        if os.path.isfile(videoPath):
            print(f"loading {videoPath} frames to RAM :")
            video = cv2.VideoCapture(os.path.abspath(videoPath)) # ouvre le fichier video comme une capture cv2
            frameReadSuccessfully, frame = video.read()
            while frameReadSuccessfully:
                # image cv2 -> conversion en surface pygame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = numpy.rot90(frame)
                pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1],"RGB")
                frame = pygame.surfarray.make_surface(frame)
                frame = pygame.transform.flip(frame, True, False)
                frame = pygame.transform.rotate(frame, angle)
                self.frames.append(frame)# sauvegarde l'image dans la RAM
                frameReadSuccessfully, frame = video.read()
                print('  processing frame #', len(self.frames), end="\r")
            video.release()

            self.frameCount = len(self.frames)
            if self.frameCount : print("\n done,",self.frameCount, "frames extracted")
            else : raise SystemError(f"error converting {videoPath} to frames")

        else : raise SystemError(f"file {videoPath} not found")

    def getFrameCount(self):
        return self.frameCount

    def getPlayingForward(self):
        return self.playingForward

    def getCurrentFrameIndex(self):
        return self.currentFrameIndex

    def getCurrentFrame(self):
        return self.frames[self.currentFrameIndex]

    def getZoom(self):
        return self.zoom

    def setZoom(self, newZoom):
        self.zoom = newZoom

    def getPoints(self):
        return self.points

    def setPoints(self, newPoints):
        self.points = newPoints

    def getConfig(self):
        return self.config

    def changeConfig(self):
        old = self.getConfig()
        if len(listPointsZoom) == 1:
            self.config = old
        else:
            while old == self.getConfig():
                self.config = random.choice([i for i in range(len(listPointsZoom))])

    def step(self, loop=False):
        """ set the play head (self.currentFrameIndex) to it's new position depending of the change direction
        proba and the current position in the file (manages looping forward or backward) """
        if loop : # entre dans la loop entre deux images définies
            if self.playingForward:
                if self.currentFrameIndex+1<=lastLoopFrame:
                    self.currentFrameIndex += 1
                else:
                    self.currentFrameIndex -= 1
                    self.playingForward = False
            else :
                if self.currentFrameIndex- 1 >= firstLoopFrame:
                    self.currentFrameIndex -= 1
                else:
                    self.currentFrameIndex += 1
                    self.playingForward = True
        else : # inverse le sens de lecture quand on atteint un bord
            if random.random() < self.changePlayDirectionProba : self.playingForward = not self.playingForward # changement de direction aléatoire
            if random.random() < fixedPlayDirectionChangeProba : self.playingForward = not self.playingForward
            if self.playingForward :
                if self.currentFrameIndex+1 < self.frameCount-1 : self.currentFrameIndex+=1
                else :
                    self.currentFrameIndex -= 1
                    self.playingForward = False
                    return
            else :
                if self.currentFrameIndex-1>=0 : self.currentFrameIndex-=1
                else :
                    self.currentFrameIndex = 1
                    self.playingForward = True
                    return

def getProbaTimer():
    """ renvoie un objet datetime de quand le prochain changement de proba est dû """
    delay = random.uniform(min(changeProbasEvery), max(changeProbasEvery))
    return datetime.now() + timedelta(seconds=delay)

def getConfigTimer():
    """ renvoie un objet datetime de quand le prochain changement de config est dû """
    delay = random.uniform(min(changeConfigEvery), max(changeConfigEvery))
    return datetime.now() + timedelta(seconds=delay)

if __name__ == "__main__":
    if len(sys.argv) < 2 : raise SystemExit(f"usage : python3 {sys.argv[0]} monFichier.mp4")
    img = ImageSequence(sys.argv[1])

    timeStarted, framesShown = datetime.now(), 0
    changeProbaTimer = getProbaTimer()
    changeConfigTimer = getConfigTimer()
    periodMillis = int(1000/maxFPS)
    deltaZoom = 0
    newZoom, oldZoom = listPointsZoom[img.getConfig()][1][0], img.getZoom()
    newPoints, oldPoints = listPointsZoom[img.getConfig()][0][0], img.getPoints()
    distance = 0
    timeTransiZoom = 0
    timeConfigStart = datetime.now()
    loop = False
    endLoop = datetime.now()
    reachZoom, reachPoints = False, False
    
    # création de la surface noir servant d'arrière plan aini que de la surface servant d'écran
    surface = pygame.display.set_mode((screenSize[0],screenSize[1]) ,pygame.FULLSCREEN) # full screen
    display = pygame.Surface((displaySize[0],displaySize[1]))
    playing = True

    def Distance(P1, P2):
        return numpy.sqrt(numpy.square(P1[0] - P2[0]) + numpy.square(P1[1] - P2[1]))

    def ZoomTranslat(image, zoomSize, point):
        # Méthode Zoom et translation dans Pygame pour la Raspberry
        wnd_w, wnd_h = displaySize[0], displaySize[1]
        image_surface = pygame.Surface((videoSize[0] / zoomSize, videoSize[1] / zoomSize))
        image_surface.blit(image, (wnd_w/(2*zoomSize) - point[0],wnd_h/(2*zoomSize) - point[1]))
        image_surface = pygame.transform.scale(image_surface, (videoSize[0], videoSize[1]))
        return image_surface


    # lire le script
    while playing :
        currentTime = datetime.now()
        
        # test de si l'on rentre dans une loop, et l'initialiser si besoin
        if random.random() < enterLoopProbas and not loop and currentTime > endLoop + timedelta(seconds=timeBetweenLoops):
            loop = True
            proba = 0
            rangeTimer = None
            drawnProba = random.random()
            for rangeProbaTimer in loopRangeProbaTimer: # choix du couple (amplitude, durée)
                proba += rangeProbaTimer[1]
                if drawnProba <= proba and rangeTimer is None:
                    rangeTimer = rangeProbaTimer
            if rangeTimer is None: # sécurité dans le cas où rangeTimer n'est pas été tiré pour une raison inconnue, cela empêche que le programme crash
                loop = False
            amplitude = numpy.random.randint(rangeTimer[0][0],rangeTimer[0][1] + 1) #on tire l'amplitude de la boucle
            loopDuration = numpy.random.uniform(rangeTimer[2][0],rangeTimer[2][1]) # on tire le temps que la boucle va durer
            shift = numpy.random.randint(rangeTimer[3][0],rangeTimer[3][1] + 1) # on tire le nb de frames dont la boucle va se déplacer
            shiftPerFrame = shift/(loopDuration*maxFPS)
            if img.getPlayingForward(): # établissement des deux bornes de la loop
                firstLoopFrame = img.getCurrentFrameIndex()
                lastLoopFrame = firstLoopFrame + amplitude
                shiftDirectionForward = True
                if lastLoopFrame >= img.getFrameCount():
                    lastLoopFrame = img.getFrameCount() - 1
                    firstLoopFrame = lastLoopFrame - amplitude
                    shiftDirectionForward = False
            else:
                lastLoopFrame = img.getCurrentFrameIndex()
                firstLoopFrame = lastLoopFrame - amplitude
                shiftDirectionForward = False
                if firstLoopFrame <= 0:
                    firstLoopFrame = 1
                    lastLoopFrame = 1 + amplitude
                    shiftDirectionForward = True
                
            endLoop = datetime.now() + timedelta(seconds=loopDuration)

        # changement de configuration
        if currentTime > changeConfigTimer and reachZoom and reachPoints:
            reachZoom = False
            reachPoints = False
            oldPoints = img.getPoints()
            oldZoom = img.getZoom()
            timeConfigStart = datetime.now()
            if random.random() < probaRandConfig: # test pour voir si l'on rentre dans une configuration tirée aléatoirement
                newZoom = numpy.random.uniform(randZoomInter[0],randZoomInter[1])
                newPoints = [numpy.random.randint(displaySize[0] / (2 * (newZoom + forkZoom)) - 1,videoSize[0] - displaySize[0] / (2 * (newZoom + forkZoom)) + 1), numpy.random.randint(displaySize[1] / (2 * (newZoom + forkZoom)) - 1, videoSize[1] - displaySize[1] / (2 * (newZoom + forkZoom)) + 1)]
                if angle == 90:
                   newPoints = [newPoints[1],videoSize[1]-newPoints[0]]
                elif angle == 180:
                    newPoints = [videoSize[0] - newPoints[0],videoSize[1] - newPoints[1]]
                elif angle == 270:
                    newPoints = [videoSize[0] - newPoints[1],newPoints[0]]
            else:
                img.changeConfig()
                if listPointsZoom[img.getConfig()][0][1] is None:
                    newPoints = listPointsZoom[img.getConfig()][0][0]
                else:
                    t = random.random()
                    X0, Y0, X1, Y1 = listPointsZoom[img.getConfig()][0][0][0], listPointsZoom[img.getConfig()][0][0][1], listPointsZoom[img.getConfig()][0][1][0], listPointsZoom[img.getConfig()][0][1][1]
                    newPoints = [int(X0 + (X1 - X0)*t), int(Y0 + (Y1 - Y0)*t)]
                if listPointsZoom[img.getConfig()][1][1] is None:
                    newZoom = listPointsZoom[img.getConfig()][1][0]
                else:
                    t = random.random()
                    newZoom = listPointsZoom[img.getConfig()][1][0] + (listPointsZoom[img.getConfig()][1][1] - listPointsZoom[img.getConfig()][1][0])*t
            distance = Distance(newPoints, oldPoints)
            timeTransiConfig = distance / avgSpeed

            # détermination si c'est le zoom ou la distance qui doit être atteint en premier
            if oldZoom > newZoom:
                timeTransiZoom = timeTransiConfig * 1.5
            else:
                timeTransiZoom = timeTransiConfig * 2/3

            changeConfigTimer = getConfigTimer()
        
        # secondes passées dans la config
        configDuration = (currentTime - timeConfigStart).total_seconds()
        
        # affichage de l'image
        if not reachZoom:
            # mise à jour de deltaZoom à chaque image
            if timeTransiZoom > configDuration:
                deltaZoom = (newZoom - img.getZoom())/(maxFPS*(timeTransiZoom - configDuration))
            else:
                deltaZoom = 0
                reachZoom = True
                
            img.setZoom(img.getZoom() + deltaZoom)
            if (img.getZoom() < newZoom + forkZoom) and (img.getZoom() >= newZoom - forkZoom):
                reachZoom = True
                changeConfigTimer = getConfigTimer()

        if not reachPoints:

            # mise à jour de deltaCols, deltaRows à chaque image
            if Distance(img.getPoints(), newPoints) != 0:
                deltaCols = numpy.sign(newPoints[0] - img.getPoints()[0]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[0] - img.getPoints()[0]) / Distance(img.getPoints(), newPoints))
                deltaRows = numpy.sign(newPoints[1] - img.getPoints()[1]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[1] - img.getPoints()[1]) / Distance(img.getPoints(), newPoints))
            else:
                deltaCols = 0
                deltaRows = 0
                reachPoints = True

            img.setPoints([img.getPoints()[0] + deltaCols, img.getPoints()[1] + deltaRows])

            if Distance(img.getPoints(), newPoints) < 5:
                reachPoints = True
                changeConfigTimer = getConfigTimer()


        display.blit(ZoomTranslat(img.getCurrentFrame(),img.getZoom(),img.getPoints()), (0,0))
        surface.blit(display,(screenSize[0]/2 - displaySize[0]/2,screenSize[1]/2 - displaySize[1]/2))
        pygame.display.update()
        for event in pygame.event.get(): # sortir avec esc
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                playing = False

        framesShown += 1 # utilisé pour calculer le framerate

        # attendre
        timeElapsed = (datetime.now() - currentTime).total_seconds()*1000 # en millis, utilisé pour calculer le retard de phase entre deux images
        if timeElapsed < periodMillis : pygame.time.wait(int(periodMillis - timeElapsed))

        # configuration de la tête de lecture pour la prochaine image
        if currentTime > changeProbaTimer :
            img.changePlayDirectionProba = random.choice(changePlayDirectionProbas)
            changeProbaTimer = getProbaTimer()
        if loop:
            if shiftDirectionForward:
                if lastLoopFrame + 1 < img.getFrameCount():
                    firstLoopFrame += shiftPerFrame
                    lastLoopFrame += shiftPerFrame
                else:
                    firstLoopFrame -= shiftPerFrame
                    lastLoopFrame -= shiftPerFrame
                    shiftDirectionForward = False
            else:
                if firstLoopFrame - 1 > 0:
                    firstLoopFrame -= shiftPerFrame
                    lastLoopFrame -= shiftPerFrame
                else:
                    firstLoopFrame += shiftPerFrame
                    lastLoopFrame += shiftPerFrame
                    shiftDirectionForward = True
            if currentTime > endLoop:
                loop = False
        img.step(loop)
        

    # calcul des FPS et on quitte le programme
    timeElapsed = (datetime.now()-timeStarted).total_seconds()
    pygame.quit()
    raise SystemExit(f"estimated FPS: {framesShown/timeElapsed:.02f}")
