import sys
import cplex
import matplotlib.pyplot as plt # For plotting solution
import math # For depot calculation if needed

TOLERANCE =10e-6 

class InstanciaRecorridoMixto:
    def __init__(self):
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
        f = open(filename)

        self.cant_clientes = int(f.readline())
        self.costo_repartidor = int(f.readline())
        self.d_max = int(f.readline())
        self.grid_length = int(f.readline()) # Leo la forma de la grilla generada

        # Leo coordenadas para plottear
        if self.grid_length > 0:
            self.depot_coord = (self.grid_length / 2.0, self.grid_length / 2.0)
        else: 
            self.depot_coord = (0.0, 0.0)


        num_refrigerados = int(f.readline())
        self.refrigerados = []
        for _ in range(num_refrigerados):
            self.refrigerados.append(int(f.readline()))

        num_exclusivos = int(f.readline())
        self.exclusivos = []
        for _ in range(num_exclusivos):
            self.exclusivos.append(int(f.readline()))

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
    for i in range(1, num_total_nodes): 
        var_names.append(f"u_{i}")
        var_types.append(prob.variables.type.continuous) 
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
    ...


def armar_lp(prob, instancia):

    escenarios = {
        'actual': modelo_actual,
        'bici': modelo_con_bici,
        #'bici_restricciones': modelo_con_bici_restricciones
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
    prob.objective.set_sense(prob.objective.sense.minimize) 

    print("Solving the problem...")
    try:
        prob.solve()
        print("Solution status: ", prob.solution.get_status_string())
        print("Solution value: ", prob.solution.get_objective_value())
    except cplex.exceptions.CplexSolverError as e:
        print(f"CPLEX Solver Error: {e}")

def plot_solution_path(instancia, solution_path, filename="solution_plot.png"):
    if not instancia.customer_coords and not instancia.depot_coord:
        print("No coordinates available to plot solution.")
        return
    
    plt.figure(figsize=(10, 10))

    all_coords = [instancia.depot_coord] + instancia.customer_coords
    
    # Plot deposito
    if instancia.depot_coord:
        plt.scatter(instancia.depot_coord[0], instancia.depot_coord[1], c='red', marker='s', s=100, label='Depósito (0)', zorder=5)
        plt.text(instancia.depot_coord[0] + 0.1, instancia.depot_coord[1] + 0.1, "0 (D)")

    # Plot clientes
    for i, coord in enumerate(instancia.customer_coords):
        client_id = i + 1 
        color = 'blue'
        marker = 'o'
        label_suffix = ""

        if client_id in instancia.refrigerados:
            color = 'cyan'
            label_suffix += "R"
        if client_id in instancia.exclusivos:
            marker = 'X' 
            color = 'magenta' if client_id in instancia.refrigerados else 'orange'
            label_suffix += "E"
        
        plt.scatter(coord[0], coord[1], c=color, marker=marker, s=70, label=f'Cliente {client_id} {label_suffix}' if i < 5 else None, zorder=5) # Label first few
        plt.text(coord[0] + 0.1, coord[1] + 0.1, str(client_id))

    # Plot camino de la solucion
    for u, v in solution_path: 
        coord_u = all_coords[u]
        coord_v = all_coords[v]
        if coord_u and coord_v:
            plt.arrow(coord_u[0], coord_u[1], coord_v[0] - coord_u[0], coord_v[1] - coord_u[1],
                      head_width=0.2, head_length=0.3, fc='gray', ec='gray', length_includes_head=True, zorder=1)

    plt.title("Ruta de la Solución")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    if instancia.grid_length > 0:
        plt.xlim(-1, instancia.grid_length + 1)
        plt.ylim(-1, instancia.grid_length + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend() # Can get crowded, enable if needed
    plt.savefig(filename)
    print(f"Gráfico de la solución guardado en: {filename}")
    plt.show()


def mostrar_solucion(prob, instancia, tolerance=1e-6):
    if prob.solution.get_status() in [prob.solution.status.optimal, 
                                      prob.solution.status.optimal_tolerance,
                                      prob.solution.status.MIP_optimal,
                                      prob.solution.status.optimal_infeasible]: 
        
        solution_values = prob.solution.get_values()
        variable_names = prob.variables.get_names()
        
        print("\nEstado de la solucion: ", prob.solution.get_status_string())
        print("Valor de la funcion objetivo: ", prob.solution.get_objective_value())
        
        print("\nVariables con valor > tolerancia (", tolerance, "):")
        active_path = []
        num_total_nodes = instancia.cant_clientes + 1

        for i in range(len(variable_names)):
            if solution_values[i] > tolerance:
                print(f"  {variable_names[i]} = {solution_values[i]}")
                if variable_names[i].startswith("x_"):
                    try:
                        parts = variable_names[i].split('_')
                        u_node = int(parts[1])
                        v_node = int(parts[2])
                        if 0 <= u_node < num_total_nodes and 0 <= v_node < num_total_nodes:
                             active_path.append((u_node, v_node))
                    except (IndexError, ValueError) as e:
                        print(f"    Warning: Could not parse path from variable {variable_names[i]}: {e}")
        
        if active_path:
            plot_filename = "solucion_ruta.png" # You can make this dynamic from input filename
            plot_solution_path(instancia, active_path, filename=plot_filename)
        else:
            print("No se encontraron arcos activos en la solución para graficar.")

    else:
        print("No se encontro una solucion optima o factible.")
        print("Estado de la solucion: ", prob.solution.get_status_string())


def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <archivo_instancia> <escenario>")
        print("Escenarios disponibles: 'actual', 'bici'")
        sys.exit(1)
    
    instancia = cargar_instancia()
    
    prob = cplex.Cplex()
    
    armar_lp(prob,instancia)

    resolver_lp(prob) 

    mostrar_solucion(prob,instancia)

if __name__ == '__main__':
    main()