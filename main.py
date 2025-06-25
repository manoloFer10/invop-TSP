import sys
import cplex
import matplotlib.pyplot as plt
import os
import time
import csv

TOLERANCE = 10e-6

class InstanciaRecorridoMixto:
    def __init__(self):
        self.filename = "" # Added to store the input filename
        self.cant_clientes = 0
        self.costo_repartidor = 0
        self.d_max = 0
        self.grid_length = 0 #exclusivo para plotteo
        self.refrigerados = []
        self.exclusivos = []
        self.customer_coords = []  #exclusivo para plotteo
        self.depot_coord = (0,0)  #exclusivo para plotteo
        
        # los indices son 0 para el deposito y 1...N para los clientes
        self.distancias = []        
        self.costos = []        

    def leer_datos(self,filename):
        self.filename = filename # Store the filename
        f = open(filename)

        self.cant_clientes = int(f.readline())
        self.costo_repartidor = float(f.readline())
        self.d_max = int(f.readline())
        self.grid_length = int(f.readline()) # Leo la forma de la grilla generada

        # Leo coordenadas para plottear
        if self.grid_length > 0:
            self.depot_coord = (self.grid_length / 2.0, self.grid_length / 2.0)
        else: 
            self.depot_coord = (0.0, 0.0)


        num_refrigerados = int(f.readline())
        self.refrigerados = [0] * (self.cant_clientes + 1)
        if num_refrigerados > 0:
            for _ in range(num_refrigerados):
                refrigerado_id = int(f.readline())
                self.refrigerados[refrigerado_id] = 1 # Indico que el cliente es refrigerado
        else:
            print("No hay clientes refrigerados.")

        num_exclusivos = int(f.readline())
        self.exclusivos = [0] * (self.cant_clientes + 1)

        if num_exclusivos > 0:
            for _ in range(num_exclusivos):
                exclusivo_id= int(f.readline())
                self.exclusivos[exclusivo_id] = 1 # Indico que el cliente es exclusivo
        else:
            print("No hay clientes exclusivos.")

        self.customer_coords = [None] * self.cant_clientes 
        for _ in range(self.cant_clientes):
            line = f.readline().split()
            client_id = int(line[0]) 
            x = float(line[1])
            y = float(line[2])
            if 1 <= client_id <= self.cant_clientes:
                self.customer_coords[client_id - 1] = (x, y)
        
        # Inicializo matrices de distancias y costos
        default_large_value = 10**9 
        num_nodes = self.cant_clientes + 1 
        self.distancias = [[default_large_value] * num_nodes for _ in range(num_nodes)]
        self.costos = [[default_large_value] * num_nodes for _ in range(num_nodes)]

        for k in range(num_nodes):
            self.distancias[k][k] = 0
            self.costos[k][k] = 0
        
        # Leo distancias cij y costos dij
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                id_i, id_j, dist_val, cost_val = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if 0 <= id_i <= self.cant_clientes and 0 <= id_j <= self.cant_clientes:
                    self.distancias[id_i][id_j] = dist_val
                    self.costos[id_i][id_j] = cost_val


        f.close()

def cargar_instancia():
    # El 1er parametro es el nombre del archivo de entrada
    nombre_archivo = sys.argv[1].strip()
    # Crea la instancia vacia
    instancia = InstanciaRecorridoMixto()
    # Llena la instancia con los datos del archivo de entrada 
    instancia.leer_datos(nombre_archivo)
    return instancia


def agregar_restricciones(prob, constraints_data):
    lin_expr = []
    senses = []
    rhs = []
    names = []

    for constr in constraints_data:
        lin_expr.append(cplex.SparsePair(ind=constr['vars'], val=constr['coeffs']))
        senses.append(constr['sense'])
        rhs.append(constr['rhs'])
        if 'name' in constr:
            names.append(constr['name'])
        else:
            names.append(f"c{len(names)}") # por default, el indice de la restriccion

    prob.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=names)
    print(f"Added {len(constraints_data)} constraints.")

def agregar_variables(prob, var_names, var_types=None, lower_bounds=None, upper_bounds=None, obj_coeffs=None):
    num_vars = len(var_names)

    if var_types is None:
        var_types = ['I'] * num_vars  # entero por default
    if lower_bounds is None:
        lower_bounds = [0.0] * num_vars
    if upper_bounds is None:
        upper_bounds = [cplex.infinity] * num_vars
    if obj_coeffs is None:
        obj_coeffs = [0.0] * num_vars
    
    prob.variables.add(obj=obj_coeffs, lb=lower_bounds, ub=upper_bounds, types=var_types, names=var_names)
    print(f"Added {num_vars} variables.")

def modelo_actual(prob, instancia):

    # Variables ============================================================================================================
    # Las defino
    var_names = []
    var_types = []
    lower_bounds = []
    upper_bounds = []
    obj_coeffs = []

    num_total_nodes = instancia.cant_clientes + 1 # Depósito (0) + N clientes

    # x_i_j variables
    for i in range(num_total_nodes):
        for j in range(num_total_nodes):
            if i == j:
                continue
            var_names.append(f"x_{i}_{j}")
            var_types.append(prob.variables.type.binary)
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)

            cost_val = instancia.costos[i][j]
            obj_coeffs.append(cost_val)


    # u_i variables 
    for i in range(0, num_total_nodes): 
        var_names.append(f"u_{i}")
        var_types.append(prob.variables.type.integer) 
        lower_bounds.append(0.0)
        upper_bounds.append(float(instancia.cant_clientes)) 
        obj_coeffs.append(0.0) # u_i no están en la f obj

    agregar_variables(prob, var_names, var_types, lower_bounds, upper_bounds, obj_coeffs)    


    # Restricciones ============================================================================================================
    constraints_data = []

    # Cada cliente es visitado exactamente una vez
    # sum_j (x_i_j) = 1 for i = 1...N
    # sum_i (x_i_j) = 1 for j = 1...N
    for k in range(1, num_total_nodes): 
        # Entra al cliente k
        vars_in = [f"x_{i}_{k}" for i in range(num_total_nodes) if i != k]
        coeffs_in = [1.0] * len(vars_in)
        constraints_data.append({'vars': vars_in, 'coeffs': coeffs_in, 'sense': 'E', 'rhs': 1.0, 'name': f"entra_a_{k}"})

        # Sale del cliente k
        vars_out = [f"x_{k}_{j}" for j in range(num_total_nodes) if j != k]
        coeffs_out = [1.0] * len(vars_out)
        constraints_data.append({'vars': vars_out, 'coeffs': coeffs_out, 'sense': 'E', 'rhs': 1.0, 'name': f"sale_de_{k}"})

    # Se empieza en el depósito
    # u_0 = 0
    # vars_depot_start = [f"u_0"]
    # coeffs_depot_start = [1.0]
    # constraints_data.append({'vars': vars_depot_start, 'coeffs': coeffs_depot_start, 'sense': 'E', 'rhs': 0.0, 'name': "inicio_deposito"})

    # El camion sale del deposito
    # sum_j (x_0_j) = 1 (para j=1..N)
    vars_depot_out = [f"x_0_{j}" for j in range(1, num_total_nodes)]
    coeffs_depot_out = [1.0] * len(vars_depot_out)
    constraints_data.append({'vars': vars_depot_out, 'coeffs': coeffs_depot_out, 'sense': 'E', 'rhs': 1.0, 'name': "sale_deposito"})
    
    # El camion regresa al deposito
    # sum_i (x_i_0) = 1 (para i=1..N)
    vars_depot_in = [f"x_{i}_0" for i in range(1, num_total_nodes)]
    coeffs_depot_in = [1.0] * len(vars_depot_in)
    constraints_data.append({'vars': vars_depot_in, 'coeffs': coeffs_depot_in, 'sense': 'E', 'rhs': 1.0, 'name': "regresa_deposito"})

    # Restricciones MTZ: u_i - u_j + N * x_i_j <= N - 1  (for i,j = 1...N, i!=j)
    N_val = float(instancia.cant_clientes)
    for i in range(1, num_total_nodes): 
        for j in range(1, num_total_nodes): 
            if i == j:
                continue
            # u_i - u_j + N * x_i_j <= N - 1
            vars_mtz = [f"u_{i}", f"u_{j}", f"x_{i}_{j}"]
            coeffs_mtz = [1.0, -1.0, N_val]
            constraints_data.append({'vars': vars_mtz, 'coeffs': coeffs_mtz, 'sense': 'L', 'rhs': N_val - 1.0, 'name': f"mtz_{i}_{j}"})
            
    #Agrego las restricciones
    agregar_restricciones(prob, constraints_data)


def modelo_con_bici(prob, instancia):
    # Variables ============================================================================================================
    # Las defino
    var_names = []
    var_types = []
    lower_bounds = []
    upper_bounds = []
    obj_coeffs = []

    num_total_nodes = instancia.cant_clientes + 1 # Depósito (0) + N clientes

    for i in range(num_total_nodes):
        for j in range(num_total_nodes):
            if i == j:
                continue
            # x_i_j variables
            var_names.append(f"x_{i}_{j}")
            var_types.append(prob.variables.type.binary)
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
            cost_val = instancia.costos[i][j]
            obj_coeffs.append(cost_val)

            # y_i_j variables
            var_names.append(f"y_{i}_{j}")
            var_types.append(prob.variables.type.binary)
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
            cost_val = instancia.costo_repartidor
            obj_coeffs.append(cost_val)

    for i in range(0, num_total_nodes): 
        # u_i variables 
        var_names.append(f"u_{i}")
        var_types.append(prob.variables.type.integer) 
        lower_bounds.append(0.0)
        upper_bounds.append(int(instancia.cant_clientes)) 
        obj_coeffs.append(0.0) # u_i no están en la f obj

        # r_i variables 
        var_names.append(f"r_{i}")
        var_types.append(prob.variables.type.integer) 
        lower_bounds.append(0.0)
        upper_bounds.append(instancia.cant_clientes) 
        obj_coeffs.append(0.0) # r_i no están en la f obj

        if i ==0 : continue

        # z_i variables 
        var_names.append(f"z_{i}")
        var_types.append(prob.variables.type.binary) 
        lower_bounds.append(0.0)
        upper_bounds.append(1.0) 
        obj_coeffs.append(0.0) # z_i no están en la f obj

        

    agregar_variables(prob, var_names, var_types, lower_bounds, upper_bounds, obj_coeffs)    


    # Restricciones ============================================================================================================
    constraints_data = []

    # Se empieza en el depósito
    # u_0 = 0
    vars_depot_start = [f"u_0"]
    coeffs_depot_start = [1.0]
    constraints_data.append({'vars': vars_depot_start, 'coeffs': coeffs_depot_start, 'sense': 'E', 'rhs': 0.0, 'name': "inicio_deposito"})

    # Se sale en camión del depósito
    # sum_j (x_0_j) = 1 (para j=1..N)
    vars_depot_out = [f"x_0_{j}" for j in range(1, num_total_nodes)]
    coeffs_depot_out = [1.0] * len(vars_depot_out)
    constraints_data.append({'vars': vars_depot_out, 'coeffs': coeffs_depot_out, 'sense': 'E', 'rhs': 1.0, 'name': "sale_deposito"})
    
    # Se llega al depo en camión
    # sum_i (x_i_0) = 1 (para i=1..N)
    vars_depot_in = [f"x_{i}_0" for i in range(1, num_total_nodes)]
    coeffs_depot_in = [1.0] * len(vars_depot_in)
    constraints_data.append({'vars': vars_depot_in, 'coeffs': coeffs_depot_in, 'sense': 'E', 'rhs': 1.0, 'name': "regresa_deposito"})

    for k in range(1, num_total_nodes): 
        # A las demás ciudades se llega en camión o bici
        vars_in = [f"x_{i}_{k}" for i in range(num_total_nodes) if i != k] + [f"y_{i}_{k}" for i in range(num_total_nodes) if i != k]
        coeffs_in = [1.0] * len(vars_in) 
        constraints_data.append({'vars': vars_in, 'coeffs': coeffs_in, 'sense': 'E', 'rhs': 1.0, 'name': f"se_llega_{k}"})

        # A la ciudad k se llega en camión
        # sum_i (x_i_j) = z_j for j = 1...N <==> sum_i (x_i_j) - z_j = 0
        vars_out = [f"x_{i}_{k}" for i in range(num_total_nodes) if i != k] + [f"z_{k}"]
        coeffs_out = [1.0] * (len(vars_out)-1) + [-1.0]
        constraints_data.append({'vars': vars_out, 'coeffs': coeffs_out, 'sense': 'E', 'rhs': 0.0, 'name': f"sale_de_{k}"})

        # Si llegué a k en camión, me fui en camión
        # z_i <= sum_j(x_i_j) <=> z_i - sum_j(x_i_j) <= 0
        vars_camion_inout = [f'z_{k}'] + [f"x_{k}_{j}" for j in range(num_total_nodes) if j != k]
        coeffs_camion_inout = [1.0] + [-1.0] * (len(vars_camion_inout)-1)
        constraints_data.append({'vars': vars_camion_inout, 'coeffs': coeffs_camion_inout, 'sense': 'L', 'rhs': 0.0, 'name': f"camion_inout_{k}"})

        #Se sale una vez de las ciudades, a lo sumo
        # sum_j (x_i_j) <= 1 for i = 1...N
        vars_out_once = [f"x_{k}_{j}" for j in range(num_total_nodes) if j != k]
        coeffs_out_once = [1.0] * len(vars_out_once)
        constraints_data.append({'vars': vars_out_once, 'coeffs': coeffs_out_once, 'sense': 'L', 'rhs': 1.0, 'name': f"sale_una_vez_de_{k}"})


    for k in range(num_total_nodes):
        # la cant de repartidores en i es mayor a la cantidad de refrigerados
        # sum_j(refrigerados_j * y_i_j) <= r_i  <==> sum_j(refrigerados_j * y_i_j) - r_i <= 0 
        vars_refrigerados = [f"y_{k}_{j}" for j in range(num_total_nodes) if j != k] + [f'r_{k}']
        coeffs_refrigerados = [instancia.refrigerados[j] for j in range(num_total_nodes) if j != k] + [-1.0] # == 0 si j es refrigerado, lista de booleanos
        constraints_data.append({'vars': vars_refrigerados, 'coeffs': coeffs_refrigerados, 'sense': 'L', 'rhs': 0.0, 'name': f"refrigerados_{k}"})

        # Cota para los repartidores
        # r_i <= sum_j(y_i_j) <==> r_i - sum_j(y_i_j) <= 0
        vars_repartidores = [f"r_{k}"] + [f"y_{k}_{j}" for j in range(num_total_nodes) if j != k]
        coeffs_repartidores = [1.0] + [-1.0] * (len(vars_repartidores)-1)
        constraints_data.append({'vars': vars_repartidores, 'coeffs': coeffs_repartidores, 'sense': 'L', 'rhs': 0.0, 'name': f"repartidores_{k}"})

        # y_i_j puede activarse si está a distancia permitida
        # c_i_j * y_i_j <= d_max
        for j in range(num_total_nodes):
            if j != 0 and j != k: # No se permite el reparto al depósito ni se reparten a sí mismos
                constraints_data.append({
                    'vars': [f"y_{k}_{j}"],
                    'coeffs': [float(instancia.distancias[k][j])],
                    'sense': 'L',
                    'rhs': float(instancia.d_max),
                    'name': f"distancia_perm_y_{k}_{j}"
                })

        for j in range(num_total_nodes):
            if k == j:
                continue
            # Si hay un reparto desde i, la cantidad de repartidores desde i siempre es >0
            # y_i_j <= r_i <==> y_i_j - r_i <= 0
            vars_repartidores_min = [f"y_{k}_{j}"] + [f"r_{k}"] 
            coeffs_repartidores_min = [1.0] + [-1.0]
            constraints_data.append({'vars': vars_repartidores_min, 'coeffs': coeffs_repartidores_min, 'sense': 'L', 'rhs': 0.0, 'name': f"repartidores_min_{k}"})

    for i in range(1, num_total_nodes):
        for j in range(1, num_total_nodes):
            if i == j: 
                continue
            # (Un repartidor sale de i entonces el camión paró en i)
            # y_i_j <= z_i <==> y_i_j - z_i <= 0
            vars_repartidores_min = [f"y_{i}_{j}"] + [f"z_{i}"] 
            coeffs_repartidores_min = [1.0] + [-1.0]
            constraints_data.append({'vars': vars_repartidores_min, 'coeffs': coeffs_repartidores_min, 'sense': 'L', 'rhs': 0.0, 'name': f"repartidores_min_{i}_{j}"})

    # Restricciones MTZ:
    for i in range(1, num_total_nodes): 
        # Se activa la pos del cliente i solo si pasa el camión
        # z_i <= u_i <= (n-1)*z_i  <=> z_i - u_i <= 0 y (n-1)*z_i - u_i >= 0
        vars_z_u = [f"z_{i}", f"u_{i}"]
        coeffs_z_u = [1.0, -1.0]
        constraints_data.append({'vars': vars_z_u, 'coeffs': coeffs_z_u, 'sense': 'L', 'rhs': 0.0, 'name': f"z_u_{i}"})
        coeffs_z_u_mtz = [int(num_total_nodes - 1), -1.0]
        constraints_data.append({'vars': vars_z_u, 'coeffs': coeffs_z_u_mtz, 'sense': 'G', 'rhs': 0.0, 'name': f"z_u_mtz_{i}"})

        # Se activa la restricción del orden de caminos si el camión pasa por las dos ciudades
        # u_i - u_j + (n-1)*x_i_j <= n-2 + (n-1)(1-z_i) + (n-1)(1-z_j)
        # <==> u_i - u_j + (n-1)*x_i_j - (n-2 + (n-1)(1-z_i) + (n-1)(1-z_j)) <= 0
        # <==> u_i - u_j + (n-1)*x_i_j - (n-1)(1-z_i) - (n-1)(1-z_j) <= n-2
        # <==> u_i - u_j + (n-1)*x_i_j + (n-1)*z_i + (n-1)*z_j <= 3n-4
        for j in range(1, num_total_nodes): 
            if i == j:
                continue
            vars_mtz = [f"u_{i}", f"u_{j}", f"x_{i}_{j}", f"z_{i}", f"z_{j}"]
            coef_mtz = [1.0, -1.0, int(num_total_nodes - 1), int(num_total_nodes - 1), int(num_total_nodes - 1)]
            constraints_data.append({'vars': vars_mtz, 'coeffs': coef_mtz, 'sense': 'L', 'rhs': 3 * num_total_nodes - 4, 'name': f"mtz_{i}_{j}"})

                        
    #Agrego las restricciones
    agregar_restricciones(prob, constraints_data)


def modelo_con_bici_entregas(prob, instancia):

    num_total_nodes = instancia.cant_clientes + 1 # Depósito (0) + N clientes

    #Instanciamos del anterior modelo
    modelo_con_bici(prob, instancia)

    #Agrego restricciones adicionales para el modelo de entregas
    constraints_data = []
    
    # Al menos cuatro entregas
    # r_i <= 4sum_i(y_i_j) <=> r_i - 4sum_i(y_i_j) <= 0
    for i in range(num_total_nodes):
        vars_entregas = [f"r_{i}"] + [f"y_{i}_{j}" for j in range(num_total_nodes) if j != i]
        coeffs_entregas = [1.0] + [-4.0] * (len(vars_entregas) - 1)
        constraints_data.append({'vars': vars_entregas, 'coeffs': coeffs_entregas, 'sense': 'L', 'rhs': 0.0, 'name': f"entregas_min_{i}"})

    agregar_restricciones(prob, constraints_data)

def modelo_con_bici_si_o_si(prob, instancia):

    num_total_nodes = instancia.cant_clientes + 1 # Depósito (0) + N clientes

    #Instanciamos del anterior modelo
    modelo_con_bici_entregas(prob, instancia)

    #Agrego restricciones adicionales para el modelo de entregas
    constraints_data = []
    
    # Exclusivos
    # z_j = 1 si instancia.exclusivos[j] == 1
    for i in range(1, num_total_nodes):
        if instancia.exclusivos[i] == 1:
            vars_exclusivo = [f"z_{i}"]
            coeffs_exclusivo = [1.0]
            constraints_data.append({'vars': vars_exclusivo, 'coeffs': coeffs_exclusivo, 'sense': 'E', 'rhs': 1.0, 'name': f"exclusivo_{i}"})

    agregar_restricciones(prob, constraints_data)

def armar_lp(prob, instancia):

    escenarios = {
        'actual': modelo_actual,
        'bici': modelo_con_bici,
        'entregas': modelo_con_bici_entregas,
        'si_o_si': modelo_con_bici_si_o_si,
    }

    escenario_elegido = sys.argv[2].strip() 
    modelo = escenarios.get(escenario_elegido, None)
    if modelo is None:
        raise ValueError(f"Modelo no reconocido. Debe ser uno de: <\n{' - '.join(escenarios.keys)}>.")

        # Agrego variables de decision y restricciones al problema

    modelo(prob, instancia)

    # Setear el sentido del problema
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Escribir el lp a archivo
    prob.write('recorridoMixto.lp')

def resolver_lp(prob): 
    # Objective sense is set before calling this function
    # prob.objective.set_sense(prob.objective.sense.minimize) 

    # Try to improve the best bound more aggressively
    prob.parameters.emphasis.mip.set(3) # 3 = MIPEmphasisBestBound

    # Increase effort for MIP heuristics (might find better integer solutions faster)
    prob.parameters.mip.strategy.heuristiceffort.set(2.0) # Default is 1.0

    # Time limit
    prob.parameters.timelimit.set(60)  # Set a time limit of 60 seconds

    print("Solving the problem...")
    start_time = time.time()
    objective_value = float('nan')
    solution_status_str = "Not Solved"
    
    try:
        prob.solve()
        solve_time = time.time() - start_time
        solution_status_str = prob.solution.get_status_string()
        
        # Check if a feasible solution was found
        if prob.solution.get_status() in [
            prob.solution.status.MIP_optimal,
            prob.solution.status.MIP_feasible,
            prob.solution.status.MIP_time_limit_feasible,
            prob.solution.status.MIP_abort_feasible
        ]:
            objective_value = prob.solution.get_objective_value()
        
        print(f"Solution status: {solution_status_str}")
        print(f"Solution value: {objective_value}")
        print(f"Solve time: {solve_time:.4f} seconds")

    except cplex.exceptions.CplexSolverError as e:
        solve_time = time.time() - start_time
        print(f"CPLEX Solver Error: {e}")
        solution_status_str = f"Solver Error: {e}"
    
    return objective_value, solve_time, solution_status_str

def plot_solution_path(instancia, truck_path, biker_path, filename="solution_plot.png"):
    if not instancia.customer_coords and not instancia.depot_coord:
        print("No coordinates available to plot solution.")
        return
    
    plt.figure(figsize=(10, 10))

    all_coords = [instancia.depot_coord] + instancia.customer_coords
    labels_added = set()

    # Plot deposito
    if instancia.depot_coord:
        plt.scatter(instancia.depot_coord[0], instancia.depot_coord[1], c='red', marker='s', s=100, label='Depósito', zorder=5)
        plt.text(instancia.depot_coord[0] + 0.1, instancia.depot_coord[1] + 0.1, "0 (D)")
        labels_added.add('Depósito')

    # Plot clientes
    for i, coord in enumerate(instancia.customer_coords):
        client_id = i + 1  
        is_refrigerated = instancia.refrigerados[client_id] == 1
        is_exclusive = instancia.exclusivos[client_id] == 1
        
        current_legend_label = None
        point_text_suffix = []

        if is_refrigerated and is_exclusive:
            color = 'magenta'
            marker = 'X'
            legend_category = "Cliente Refrigerado y Exclusivo"
            if is_refrigerated: point_text_suffix.append("R")
            if is_exclusive: point_text_suffix.append("E")
        elif is_refrigerated:
            color = 'cyan'
            marker = 'o'
            legend_category = "Cliente Refrigerado"
            point_text_suffix.append("R")
        elif is_exclusive:
            color = 'orange'
            marker = 'X'
            legend_category = "Cliente Exclusivo"
            point_text_suffix.append("E")
        else: # Regular client
            color = 'blue'
            marker = 'o'
            legend_category = "Cliente"
        
        if legend_category not in labels_added:
            current_legend_label = legend_category
            labels_added.add(legend_category)
        
        plt.scatter(coord[0], coord[1], c=color, marker=marker, s=70, label=current_legend_label, zorder=5)
        suffix_str = ",".join(point_text_suffix)
        plt.text(coord[0] + 0.1, coord[1] + 0.1, f"{client_id}{f' ({suffix_str})' if suffix_str else ''}")

    # Plot truck path
    truck_label_added = False
    for u, v in truck_path: 
        coord_u = all_coords[u]
        coord_v = all_coords[v]
        if coord_u and coord_v:
            label_to_use = None
            if not truck_label_added:
                label_to_use = 'Ruta Camión'
                truck_label_added = True
            plt.arrow(coord_u[0], coord_u[1], coord_v[0] - coord_u[0], coord_v[1] - coord_u[1],
                      head_width=0.2, head_length=0.3, fc='gray', ec='gray', length_includes_head=True, zorder=3, label=label_to_use)

    # Plot biker path
    biker_label_added = False
    for u, v in biker_path:
        coord_u = all_coords[u]
        coord_v = all_coords[v]
        if coord_u and coord_v:
            label_to_use = None
            if not biker_label_added:
                label_to_use = 'Ruta Repartidor'
                biker_label_added = True
            plt.arrow(coord_u[0], coord_u[1], coord_v[0] - coord_u[0], coord_v[1] - coord_u[1],
                      head_width=0.2, head_length=0.3, fc='green', ec='green', linestyle='dashed', length_includes_head=True, zorder=2, label=label_to_use)


    plt.title("Ruta de la Solución")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    if instancia.grid_length > 0:
        plt.xlim(-1, instancia.grid_length + 1)
        plt.ylim(-1, instancia.grid_length + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    
    # Filter out None labels before creating legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels if any, keeping the first instance
    if by_label: # Check if there's anything to show in the legend
        plt.legend(by_label.values(), by_label.keys())

    plt.savefig(filename)
    print(f"Gráfico de la solución guardado en: {filename}")
    plt.show()


def mostrar_solucion(prob, instancia, output_folder_path, scenario_name, tolerance=1e-6):
    solution_status_code = prob.solution.get_status()
    
    plotworthy_statuses = [
        prob.solution.status.MIP_optimal,
        prob.solution.status.MIP_feasible,
        prob.solution.status.MIP_time_limit_feasible,
        prob.solution.status.MIP_abort_feasible
    ]

    base_instance_name = os.path.splitext(os.path.basename(instancia.filename))[0]
    plot_filename = os.path.join(output_folder_path, f"{base_instance_name}_{scenario_name}_solution.png")

    if solution_status_code in plotworthy_statuses: 
        solution_values = prob.solution.get_values()
        variable_names = prob.variables.get_names()
        
        print(f"\nScenario: {scenario_name}")
        print("Estado de la solucion: ", prob.solution.get_status_string())
        print("Valor de la funcion objetivo: ", prob.solution.get_objective_value())
        
        print("\nVariables con valor > tolerancia (", tolerance, ") o variables de estado (u_, z_, r_):")
        truck_path = []
        biker_path = []
        num_total_nodes = instancia.cant_clientes + 1

        for i in range(len(variable_names)):
            var_name = variable_names[i]
            var_value = solution_values[i]

            if var_value > tolerance or var_name.startswith("u_") or var_name.startswith("z_") or var_name.startswith("r_"):
                print(f"  {var_name} = {var_value}")

                if var_name.startswith("x_") and var_value > tolerance:
                    try:
                        parts = var_name.split('_')
                        u_node = int(parts[1])
                        v_node = int(parts[2])
                        if 0 <= u_node < num_total_nodes and 0 <= v_node < num_total_nodes:
                             truck_path.append((u_node, v_node))
                        else:
                            print(f"    Warning: Parsed node ID out of range for x_ variable. u={u_node}, v={v_node}. Max expected index is {num_total_nodes - 1}.")
                    except (IndexError, ValueError) as e:
                        print(f"    Warning: Could not parse path from variable {var_name}: {e}")
                elif var_name.startswith("y_") and var_value > tolerance:
                    try:
                        parts = var_name.split('_')
                        u_node = int(parts[1])
                        v_node = int(parts[2])
                        if 0 <= u_node < num_total_nodes and 0 <= v_node < num_total_nodes:
                             biker_path.append((u_node, v_node))
                        else:
                            print(f"    Warning: Parsed node ID out of range for y_ variable. u={u_node}, v={v_node}. Max expected index is {num_total_nodes - 1}.")
                    except (IndexError, ValueError) as e:
                        print(f"    Warning: Could not parse path from variable {var_name}: {e}")
        
        if truck_path or biker_path:
            # The plot_filename is now constructed above and passed correctly
            plot_solution_path(instancia, truck_path, biker_path, filename=plot_filename)
        else:
            print("No se encontraron arcos activos (x_i_j > tolerance o y_i_j > tolerance) en la solución para graficar.")

    else:
        print(f"\nScenario: {scenario_name}")
        print("No se encontro una solucion optima o factible para graficar.")
        print("Estado de la solucion: ", prob.solution.get_status_string())
        try:
            print("Valor (si disponible): ", prob.solution.get_objective_value())
        except cplex.exceptions.CplexError:
            pass

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <archivo_instancia>")
        sys.exit(1)
    
    instance_file_path = sys.argv[1].strip()
    
    instancia = cargar_instancia() # This will also set instancia.filename via leer_datos

    base_instance_name = os.path.splitext(os.path.basename(instance_file_path))[0]
    output_folder_path = f"{base_instance_name}_results"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")

    escenarios_map = {
        'actual': modelo_actual,
        'bici': modelo_con_bici,
        'entregas': modelo_con_bici_entregas,
        'si_o_si': modelo_con_bici_si_o_si,
    }

    summary_data = []

    for scenario_name, model_function in escenarios_map.items():
        print(f"\n--- Running scenario: {scenario_name} ---")
        prob = cplex.Cplex()
        
        # Call the specific model function to add variables and constraints
        model_function(prob, instancia)

        # Set objective sense (common for all models)
        prob.objective.set_sense(prob.objective.sense.minimize)

        # Write the LP file for the current scenario
        lp_filename = os.path.join(output_folder_path, f"{base_instance_name}_{scenario_name}.lp")
        try:
            prob.write(lp_filename)
            print(f"LP model for scenario {scenario_name} written to {lp_filename}")
        except cplex.exceptions.CplexError as e:
            print(f"Error writing LP file for {scenario_name}: {e}")

        # Solve the problem
        obj_val, s_time, sol_status = resolver_lp(prob)
        
        summary_data.append({
            'Scenario': scenario_name,
            'ObjectiveValue': obj_val if not isinstance(obj_val, float) or (not cplex.infinity == obj_val and not (obj_val != obj_val)) else str(obj_val), # handle NaN and Inf for CSV
            'SolveTimeSeconds': f"{s_time:.4f}",
            'SolutionStatus': sol_status
        })

        # Display solution and plot (if feasible)
        mostrar_solucion(prob, instancia, output_folder_path, scenario_name)
        
        # Clean up CPLEX problem object for the next iteration
        del prob

    # Write summary CSV
    csv_file_path = os.path.join(output_folder_path, f"{base_instance_name}_summary.csv")
    csv_columns = ['Scenario', 'ObjectiveValue', 'SolveTimeSeconds', 'SolutionStatus']
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data_row in summary_data:
                writer.writerow(data_row)
        print(f"\nSummary written to {csv_file_path}")
    except IOError:
        print(f"I/O error writing CSV to {csv_file_path}")

if __name__ == '__main__':
    main()