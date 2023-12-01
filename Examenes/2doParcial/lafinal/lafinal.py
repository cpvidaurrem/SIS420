# -*- coding: utf-8 -*-
import sys
import pygame
import numpy as np
import random as rnd
from pygame.locals import *

import os

import matplotlib.pyplot as plt
import random

# Ventana = 9 x 7 recuadros
#
# B = Balón
# A = Arquero
#
#
# (1,1)
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |===|===| A |===|===|   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   | B |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#                                       (9,7)
#
#


#######################
# Clases  y funciones #
#######################

def pos_recuadro(x, y):
    new_x = x
    new_y = y
    # Ajustar a los limites
    if (x < 1):
        new_x = 1
    if (x > 9):
        new_x = 9
    if (y < 1):
        new_y = 1
    if (y > 7):
        new_y = 9
    return ((new_x-1) * PPR, (new_y-1) * PPR)

def mover_x(pos, desplazamiento):
    return (pos[0] + desplazamiento, pos[1])

def mover_y(pos, desplazamiento):
    return (pos[0], pos[1] + desplazamiento)


class Balon:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recuadro = rnd.randint(1,9)
        self.x = pos_recuadro(self.recuadro, 7)[0]
        self.y = pos_recuadro(self.recuadro, 7)[1]

    def pos(self):
        return (self.x, self.y)

    def avanzar(self, factor):
        if (self.y > 0):
            nuevo_y = self.y - factor
            if nuevo_y <= 0:
                self.y = 0
            else:
                self.y = nuevo_y

    def esta_fuera(self):
        return (self.recuadro <= 2 or self.recuadro >= 8)

    def esta_en_red(self):
        return (self.recuadro >= 3 and self.recuadro <= 7)


class Arquero:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = ARQUERO_POS_INICIAL[0]
        self.y = ARQUERO_POS_INICIAL[1]
        self.recuadro = 5

    def pos(self):
        return (self.x, self.y)

    def esta_en_red(self):
        return self.recuadro >= 3 and self.recuadro <= 7

    def pasos_izquierda(self, pasos):
        if (self.recuadro > 1):
            if (self.recuadro - pasos >= 1):
                self.recuadro -= pasos
            else:
                self.recuadro = 1
            self.x = pos_recuadro(self.recuadro, 1)[0]

    def pasos_derecha(self, pasos):
        if (self.recuadro < 9):
            if (self.recuadro + pasos <= 9):
                self.recuadro += pasos
            else:
                self.recuadro = 9
            self.x = pos_recuadro(self.recuadro, 1)[0]

    # Aplicar un valor numérico de acción (entre 0 y 18) como un movimiento
    # Las acciones son pasos del arquero (19 acciones posibles)
    # movimientos negativos son a la izquierda, y positivos son a la derecha
    #  acción     =  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18
    #  movimiento = -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1   2   3   4   5   6   7   8   9
    def accion_a_movimiento(self, accion): #
        accion_calculada = accion - 9 #
        if (accion_calculada == 0): # No moverse
            return
        elif (accion_calculada < 0): # Moverse a la izquierda
            self.pasos_izquierda(abs(accion_calculada))
        elif (accion_calculada > 0): # Moverse a la derecha
            self.pasos_derecha(abs(accion_calculada))




class Estado:
    def __init__(self, arquero, balon):
        self.arquero = arquero.recuadro
        self.balon = balon.recuadro

    def get_rep(self):
        return str(self.arquero) + str(self.balon)


def reset():
    global balon, arquero, CAPTURAS, PUNTOS, GOLES, FUERA, EPISODIOS
    CAPTURAS = PUNTOS = GOLES = FUERA = EPISODIOS = 0
    arquero.reset()
    balon.reset()
    print("\n#  RESET\n")



##############
# Constantes #
##############

RECUADROS_ANCHO = 9
RECUADROS_ALTO  = 7
PIXELES_POR_RECUADRO = PPR =  100

VENTANA_ANCHO = PPR * RECUADROS_ANCHO
VENTANA_ALTO  = PPR * RECUADROS_ALTO
VENTANA_COLOR = (10,175,30)

FPS = 60
FACTOR_BALON = 5

ARQUERO_POS_INICIAL = pos_recuadro(5, 1)
BALON_POS_INICIAL = pos_recuadro(5, 7)
RED_POS_INICIAL = pos_recuadro(3, 1)




#################
# Estado Global #
#################

CAPTURAS      = 0
GOLES         = 0
FUERA         = 0
PUNTOS        = 0
EPISODIOS     = 0 # Número de episodios

 # Parámetros de aprendizaje
EPSILON       = 0.9 # Probabilidad de exploración
GAMMA         = 0.1  # Factor de descuento 
LEARNING_RATE = 1.0  # Tasa de aprendizaje 




# Matriz Q como diccionario de estados a vectores de acciones
#
# - Los estados son instancias de la clase `Estado`
# - Las acciones son pasos del arquero (19 acciones posibles)
#     -  0-8  = pasos a la izquierda
#     -   9   = no moverse
#     - 10-18 = pasos a la derecha
Q = {} # Diccionario vacío





# Escoger la mejor acción disponible para un estado particular
# def mejor_accion(estado):
#     if not (estado in Q) :        # Aun no existe este estado
#         Q[estado] = [0] * 19     # Inicializar con 0 para todas las acciones
#         return rnd.randint(0,18) # Retornar una acción aleatoria
#     else:
#         return Q[estado].index(max(Q[estado]))
    
def mejor_accion(estado):
    if not (estado in Q):  # Aun no existe este estado
        Q[estado] = [0] * 19  # Inicializar con 0 para todas las acciones
        #return rnd.randint(0, 18)  # Retornar una acción aleatoria

    if random.random() > EPSILON:
        # Exploración: Tomar una acción aleatoria
        return rnd.randint(0, 18)
    else:
        # Explotación: Tomar la mejor acción conocida
        return Q[estado].index(max(Q[estado])) # Retorna el índice de la acción con mayor valor en la matriz Q





########################################
# Ventana, imágenes y objetos de juego #
########################################

fpsClock = pygame.time.Clock()
pygame.init()
screen = pygame.display.set_mode((VENTANA_ANCHO, VENTANA_ALTO))
pygame.display.set_caption("La Final")

balon_img = os.path.join(os.path.dirname(__file__), 'img')
balon_img = pygame.image.load(os.path.join(balon_img, 'balon.png'))


#balon_img = pygame.image.load('img/balon.png')
balon_img = pygame.transform.scale(balon_img, (PPR, PPR))

arquero_img = os.path.join(os.path.dirname(__file__), 'img')
arquero_img = pygame.image.load(os.path.join(arquero_img, 'arquero.png'))

#arquero_img = pygame.image.load('img/arquero.png')

arquero_img = pygame.transform.scale(arquero_img, (PPR, PPR))

red_img = os.path.join(os.path.dirname(__file__), 'img')
red_img = pygame.image.load(os.path.join(red_img, 'red.png'))
#red_img = pygame.image.load('img/red.png')

red_img = pygame.transform.scale(red_img, (5*PPR, PPR))

font = pygame.font.Font(None, 30)
font_small = pygame.font.Font(None, 20)

balon = Balon()
arquero = Arquero()



puntaje_acumulado_por_episodio = []
numero_de_episodios = []

######################
# Bucle de Episodios #
######################


while True:
    balon.reset()
    EPISODIOS += 1

    print("NRO EPISODIO: " + str(EPISODIOS) + " " + " PUNTAJE: " + str(PUNTOS))

    #################
    # Elegir acción #
    #################
    estado = Estado(arquero, balon).get_rep()
    print("Estado: " + estado)
    accion = mejor_accion(estado)
    print("Accion: " + str(accion))
    arquero.accion_a_movimiento(accion)
    nuevo_estado = Estado(arquero, balon).get_rep()
    print("Nuevo Estado: " + nuevo_estado)



    while (balon.y > 0): # Mientras el balón no haya llegado al piso
        for event in pygame.event.get():

            if (event.type == QUIT):
                pygame.quit()
                sys.exit()
            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_r):
                    reset()
                if (event.key == pygame.K_SPACE): # Turbo
                    if (FPS == 60):
                        FPS = 1000
                        FACTOR_BALON = 100
                    else:
                        FPS = 60
                        FACTOR_BALON = 5
                if (event.key == pygame.K_RIGHT):
                    arquero.pasos_derecha(1)
                if (event.key == pygame.K_LEFT):
                    arquero.pasos_izquierda(1)
                if (event.key == pygame.K_q):
                    exit()




        #################
        # Avanzar juego #
        #################
    
        balon.avanzar(FACTOR_BALON) 
        screen.fill(VENTANA_COLOR)
        screen.blit(red_img,     RED_POS_INICIAL)
        screen.blit(arquero_img, arquero.pos())
        screen.blit(balon_img,   balon.pos())

        goles_texto = font.render('Goles: ' + str(GOLES), True, (255, 255, 255))
        fuera_texto = font.render('Fuera: ' + str(FUERA), True, (255, 255, 255))
        capturas_texto = font.render('Tapadas: ' + str(CAPTURAS), True, (255, 255, 255))
        puntos_texto = font.render('Puntos: ' + str(PUNTOS), True, (255, 255, 255))
        episodios_texto = font.render('Penales: ' + str(EPISODIOS), True, (255, 255, 255))
        screen.blit(goles_texto, (20, VENTANA_ALTO - 30))
        screen.blit(fuera_texto, (180, VENTANA_ALTO - 30))
        screen.blit(capturas_texto, (VENTANA_ANCHO - 350, VENTANA_ALTO - 30))
        screen.blit(puntos_texto, (VENTANA_ANCHO - 150, VENTANA_ALTO - 30))
        screen.blit(episodios_texto, (VENTANA_ANCHO - 550, VENTANA_ALTO - 30))

        pygame.display.update()
        fpsClock.tick(FPS)





        ########################
        # Reglas de recompensa #
        ########################

        recompensa = 0
        # Balón fuera de la red
        if (balon.esta_fuera()):
            if (arquero.esta_en_red()):
                recompensa += 1 # Regla 4
            else:
                recompensa -= 1 # Regla 3

        # Balón capturado
        elif (balon.esta_en_red() and arquero.esta_en_red() \
                and (balon.recuadro == arquero.recuadro)):
            recompensa += 2 # Regla 1

        # Gol
        elif (balon.esta_en_red() and (balon.recuadro != arquero.recuadro)):
            recompensa -= 2 # Regla 2





        ##############
        # Aprender Q #
        ##############
        if not (nuevo_estado in Q):
            Q[nuevo_estado] = [0] * 19
        Q[estado][accion] += LEARNING_RATE * (recompensa + GAMMA * max(Q[nuevo_estado]) - Q[estado][accion]) 




    ## Actualizar información en patanlla
    # Balón fuera de la red
    if (balon.esta_fuera()):
        FUERA += 1
        if (arquero.esta_en_red()):
            PUNTOS += 1 # Regla 4
        else:
            PUNTOS -= 1 # Regla 3

    # Balón capturado
    elif (balon.esta_en_red() and arquero.esta_en_red() \
            and (balon.recuadro == arquero.recuadro)):
        CAPTURAS += 1
        PUNTOS += 2 # Regla 1

    # Gol
    elif (balon.esta_en_red() and (balon.recuadro != arquero.recuadro)):
        GOLES += 1
        PUNTOS -= 2 # Regla 2



    # Graficar curva de aprendizaje
    
    puntaje_acumulado_por_episodio.append(PUNTOS)
    numero_de_episodios.append(EPISODIOS)

    if (EPISODIOS == 2500):
        # Al final del programa, crea la curva de aprendizaje
        plt.figure()
        plt.plot(numero_de_episodios, puntaje_acumulado_por_episodio)
        plt.title("Curva de Aprendizaje Q-Learning")
        plt.xlabel("Número de Episodios")
        plt.ylabel("Puntaje Acumulado")
        plt.show()
        
        print("Matriz Q")
        print(Q)

        # Al final del programa, crea la matriz Q
        plt.figure()
        plt.imshow(np.array([Q[key] for key in sorted(Q.keys())]))
        plt.title("Matriz Q")
        plt.xlabel("Acciones")
        plt.ylabel("Estados")
        plt.show()