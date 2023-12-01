"""
Un ejemplo simple de aprendizaje por refuerzo utilizando el método Q-learning de búsqueda de tablas.
Un agente "o" está a la izquierda de un mundo unidimensional, el tesoro está en el lugar más a la derecha.
Ejecute este programa y vea cómo el agente mejorará su estrategia para encontrar el tesoro

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

"""
T--o------Tarea
"""

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

np.random.seed(2)  # reproducible


N_STADOS = 10   # La longitud del mundo unidimensional.
ACTIONS = ['left', 'right']     # acciones disponibles
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate(tasa de aprendisaje)
GAMMA = 0.9    # factor de descuento
MAX_EPISODIOS = 15   # episodios máximos
NUEVO_TIME = 0.1    # nuevo tiempo para un movimiento

puntaje_acumulado_por_episodio = []
numero_de_episodios = []


def Crear_q_table(n_stados, actions):
    table = pd.DataFrame(
        np.zeros((n_stados, len(actions))),     # q_table valores iniciales
        columns=actions,    # nombre de las acciones
    )
    # print(table)    # mostrar tabla
    return table


def eligir_action(stados, q_table): 
    # Así es como elegir una acción.
    stados_actions = q_table.iloc[stados, :] # seleccionar la accion de un estado en particular 
    if (np.random.uniform(0,1) < EPSILON) or ((stados_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS) # elegir una acción aleatoria 
    else:   # act greedy
        action_name = stados_actions.idxmax()  # elegir la acción con el valor máximo (si hay varios, elija el primero)
    return action_name # devuelve la acción elegida 


def get_entorno_feedback(S, A): # S = estado, A = accion 
    # Así es como el agente interactuará con el entorno.
    if A == 'right':    # mover a la derecha
        if S == N_STADOS - 2:  # el estado más a la derecha
            S_ = 'terminal'  # terminal
            R = 1 # recompensa
        else:
            S_ = S + 1
            R = 0
    else:   # mover hacia la izquierda
        R = 0 
        if S == 0:
            S_ = S  # llegar a la pared
        else:
            S_ = S - 1
    return S_, R


def actualizar_entorn(S, episodio, pasos_contadr): # 
    # Así se actualiza el entorno
    #print('_______________________________')
    salto_vac = '                                      '
    entorno_list =['T']+['-']*(N_STADOS-1) + ['Tarea']   # '---------T' our environment
    if S == 'terminal':
        interaction ='Episode %s:total_pasos = %s' % (episodio+1, pasos_contadr)
       
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        entorno_list[S] = 'o'
        interaction = ''.join(entorno_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(NUEVO_TIME)
        

def Aprendizje_Refuerzo ():
    # main part of RL loop
    q_table = Crear_q_table(N_STADOS, ACTIONS)
    
  
    
    for episodio in range(MAX_EPISODIOS):
        pasos_contadr = 0
        S = 0
        is_terminated = False
        actualizar_entorn(S, episodio, pasos_contadr)
        while not is_terminated:

            A = eligir_action(S, q_table) # elegir una acción para este estado 
            next_Stado, Reconp = get_entorno_feedback(S, A)  # tomar acción y obtener el siguiente estado y la recompensa
            q_predict = q_table.loc[S, A] # obtener el valor de q_table en el estado actual y acción actual 
            if next_Stado != 'terminal': # si el siguiente estado no es terminal
                q_objetiv = Reconp + GAMMA * (q_table.iloc[next_Stado, :].max())  
            else: # si el siguiente estado es terminal
                q_objetiv = Reconp     # q_objetivo = r (sin estado siguiente, esto es todo)
                is_terminated = True    # terminar este episodio

            q_table.loc[S, A] += ALPHA * (q_objetiv - q_predict)  # actualizar q_table
            S = next_Stado  # mover al siguiente estado

            actualizar_entorn(S, episodio, pasos_contadr+1) # actualizar el entorno
            pasos_contadr += 1 # actualizar el contador de pasos

    return q_table # devolver la tabla Q




            # EPISODIOS = episodio
            # PUNTOS = pasos_contadr
            # if (EPISODIOS <= MAX_EPISODIOS):
            #     puntaje_acumulado_por_episodio.append(PUNTOS)
            #     numero_de_episodios.append(EPISODIOS)

                
    # Al final del programa, crea la curva de aprendizaje
    # plt.figure()
    # plt.plot(numero_de_episodios, puntaje_acumulado_por_episodio)
    # plt.title("Curva de Aprendizaje Q-Learning")
    # plt.xlabel("Número de Episodios")
    # plt.ylabel("Puntaje Acumulado")
    # plt.show()

    # # Graficar la tabla Q
    # plt.figure(figsize=(8, 6))
    # plt.imshow(q_table, cmap='coolwarm')
    # plt.colorbar()
    # plt.xticks(np.arange(len(ACTIONS)), ACTIONS)
    # plt.yticks(np.arange(N_STADOS))
    # plt.xlabel('Acciones')
    # plt.ylabel('Estados')
    # plt.title('Tabla Q')
    # plt.show()
    
    #return q_table

    


if __name__ == "__main__":
    print('\n\n\n')

    q_table = Aprendizje_Refuerzo ()
    print('\r\nQ-table:\n')
    print(q_table)

    q_table_e = q_table[:-1]

    valores_redondeados=[round(q_table_e,3)]
    print(valores_redondeados)
    
    plt.figure()
    plt.plot(q_table_e)
    plt.title("Curva de Aprendizaje Q-Table")
    plt.xlabel("Número de Stados")
    plt.ylabel("Acciones")
    plt.show()
    plt.show()