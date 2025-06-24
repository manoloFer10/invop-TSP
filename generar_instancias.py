import random
import math
import sys
import matplotlib.pyplot as plt

# Set a seed for reproducibility (optional, can be set once at the start)
RANDOM_SEED = 42 
random.seed(RANDOM_SEED)

def manhattan_distance(p1, p2):
    """Calculates Manhattan distance between two points (tuples or lists)."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def plot_instancia(client_coords, refrigerados_ids, exclusivos_ids, grid_length, filename="instancia_plot.png"):
    """
    Plots the client locations.
    Args:
        client_coords (list): List of (x,y) tuples for each client.
        refrigerados_ids (list): List of 1-based IDs for refrigerated clients.
        exclusivos_ids (list): List of 1-based IDs for exclusive (truck) clients.
        grid_length (int): The side length of the grid.
        filename (str): Filename to save the plot.
    """
    if not client_coords:
        print("No clients to plot.")
        return

    plt.figure(figsize=(8, 8))
    
    x_coords = [c[0] for c in client_coords]
    y_coords = [c[1] for c in client_coords]

    # Plot all clients initially
    plt.scatter(x_coords, y_coords, c='blue', label='Cliente Regular', s=50)

    # Highlight refrigerated clients
    refrigerados_coords_x = []
    refrigerados_coords_y = []
    for client_id in refrigerados_ids:
        if 0 < client_id <= len(client_coords):
            refrigerados_coords_x.append(client_coords[client_id-1][0])
            refrigerados_coords_y.append(client_coords[client_id-1][1])
    if refrigerados_coords_x:
        plt.scatter(refrigerados_coords_x, refrigerados_coords_y, c='cyan', label='Cliente Refrigerado', s=70, edgecolors='black')

    # Highlight exclusive clients
    exclusivos_coords_x = []
    exclusivos_coords_y = []
    for client_id in exclusivos_ids:
        if 0 < client_id <= len(client_coords):
            exclusivos_coords_x.append(client_coords[client_id-1][0])
            exclusivos_coords_y.append(client_coords[client_id-1][1])
    if exclusivos_coords_x: # Check if there are any exclusive clients to plot
        # Use a different marker or color for exclusive, especially if they can also be refrigerated
        # For simplicity, let's assume exclusive overrides regular/refrigerated for plotting color if it's also refrigerated
        # Or, we can plot them with a different marker
        plt.scatter(exclusivos_coords_x, exclusivos_coords_y, c='red', marker='X', label='Cliente Exclusivo (Camión)', s=100)


    for i, (x, y) in enumerate(client_coords):
        plt.text(x + 0.1, y + 0.1, str(i + 1))  # Client ID (1-based)

    plt.title(f"Distribución de Clientes (Semilla: {RANDOM_SEED})")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.xlim(-1, grid_length)
    plt.ylim(-1, grid_length)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    print(f"Gráfico de la instancia guardado en: {filename}")
    # plt.show() # Uncomment to display the plot directly

def generar_instancia(filepath, num_total_clientes, grid_length, plot=True):
    """
    Generates a problem instance and saves it to a file.

    Args:
        filepath (str): Path to the output file.
        num_total_clientes (int): The total number of clients.
        grid_length (int): The side length of the grid for placing clients.
                           Must be such that grid_length*grid_length >= num_total_clientes.
        plot (bool): Whether to generate and save a plot of the instance.
    """
    # Seed is set globally, but could also be set here if preferred per-generation
    # random.seed(RANDOM_SEED) 

    if num_total_clientes < 0:
        print("Error: Number of clients cannot be negative.")
        return
    if num_total_clientes > 0 and grid_length * grid_length < num_total_clientes:
        print(f"Error: Grid size {grid_length}x{grid_length} is too small for {num_total_clientes} clients.")
        return
    if num_total_clientes > 0 and grid_length <= 0:
        print(f"Error: Grid length must be positive for num_total_clientes > 0.")
        return

    # 1. Cantidad de clientes
    cant_clientes = num_total_clientes

    # 2. Costo repartidor
    costo_repartidor = random.randint(15, 75)

    # Client coordinates
    client_coords_list = [] # List of (x,y) tuples, 0-indexed for internal use here
    if cant_clientes > 0:
        all_possible_points = [(x, y) for x in range(grid_length) for y in range(grid_length)]
        random.shuffle(all_possible_points)
        client_coords_list = all_possible_points[:cant_clientes]

    # 3. Distancia máxima repartidor
    dist_max = 1 
    if cant_clientes > 1 and grid_length > 0:
        min_dist_max_val = max(1, int(grid_length * 0.1))
        max_dist_max_val = max(1, int(grid_length * 0.5))
        if min_dist_max_val > max_dist_max_val:
             max_dist_max_val = min_dist_max_val
        dist_max = random.randint(min_dist_max_val, max_dist_max_val)
    elif cant_clientes == 1:
        dist_max = 1 

    all_client_ids = list(range(1, cant_clientes + 1)) 

    # 4. Clientes refrigerados
    num_refrigerados = 0
    if cant_clientes > 0:
        min_r_pct = 0.05
        max_r_pct = 0.30
        min_r = int(cant_clientes * min_r_pct)
        max_r = int(cant_clientes * max_r_pct)
        if max_r == 0 and cant_clientes > 0 and max_r_pct > 0: max_r = 1
        if min_r > max_r: min_r = max_r
        num_refrigerados = random.randint(min_r, max_r)
    
    refrigerados_ids = []
    if num_refrigerados > 0:
        refrigerados_ids = sorted(random.sample(all_client_ids, min(num_refrigerados, cant_clientes)))

    # 5. Clientes exclusivos (camión)
    num_exclusivos = 0
    if cant_clientes > 0:
        min_e_pct = 0.05
        max_e_pct = 0.30
        min_e = int(cant_clientes * min_e_pct)
        max_e = int(cant_clientes * max_e_pct)
        if max_e == 0 and cant_clientes > 0 and max_e_pct > 0: max_e = 1
        if min_e > max_e: min_e = max_e
        num_exclusivos = random.randint(min_e, max_e)

    exclusivos_ids = []
    if num_exclusivos > 0:
        # Ensure exclusive clients are distinct from each other, but can overlap with refrigerated
        available_for_exclusive = list(set(all_client_ids)) # All clients are candidates
        exclusivos_ids = sorted(random.sample(available_for_exclusive, min(num_exclusivos, len(available_for_exclusive))))
    
    # 6. Distancias (cij) y costos de camión (dij)
    distancias_costos_lines = []
    if cant_clientes > 1:
        truck_cost_multiplier = random.uniform(1.0, 3.0) 
        for i in range(cant_clientes):
            for j in range(i + 1, cant_clientes):
                client_id_i = i + 1 
                client_id_j = j + 1 
                
                coord_i = client_coords_list[i]
                coord_j = client_coords_list[j]
                
                dist_ij = manhattan_distance(coord_i, coord_j)
                if dist_ij == 0: dist_ij = 1 
                
                cost_ij = int(dist_ij * truck_cost_multiplier)
                if cost_ij < 1: cost_ij = 1
                
                distancias_costos_lines.append(f"{client_id_i} {client_id_j} {dist_ij} {cost_ij}")

    # Write to file
    with open(filepath, 'w') as f:
        f.write(f"{cant_clientes}\n")
        f.write(f"{costo_repartidor}\n")
        f.write(f"{dist_max}\n")
        f.write(f"{grid_length}\n") # <-- NEW: Save grid_length

        f.write(f"{len(refrigerados_ids)}\n")
        for client_id in refrigerados_ids:
            f.write(f"{client_id}\n")
            
        f.write(f"{len(exclusivos_ids)}\n")
        for client_id in exclusivos_ids:
            f.write(f"{client_id}\n")

        # <-- NEW: Save client coordinates -->
        # Assuming client_coords_list is 0-indexed internally but we write 1-based IDs
        if cant_clientes > 0:
            for i in range(cant_clientes):
                client_id = i + 1
                x, y = client_coords_list[i]
                f.write(f"{client_id} {x} {y}\n")
        # <-- End of new client coordinates section -->
            
        for line in distancias_costos_lines:
            f.write(f"{line}\n")

    print(f"Instancia generada con {cant_clientes} clientes y guardada en: {filepath}")
    if cant_clientes > 0:
        print(f"  Semilla aleatoria usada: {RANDOM_SEED}")
        print(f"  Costo repartidor: {costo_repartidor}")
        print(f"  Distancia máxima repartidor: {dist_max}")
        print(f"  Clientes refrigerados ({len(refrigerados_ids)}): {refrigerados_ids if refrigerados_ids else 'Ninguno'}")
        print(f"  Clientes exclusivos ({len(exclusivos_ids)}): {exclusivos_ids if exclusivos_ids else 'Ninguno'}")
        # print(f"  Coordenadas de clientes (ej. primeros 5): {client_coords_list[:5]}")
        # print(f"  Líneas de distancia/costo generadas: {len(distancias_costos_lines)}")

    if plot and cant_clientes > 0:
        plot_filename = filepath.replace(".txt", "") + "_plot.png"
        plot_instancia(client_coords_list, refrigerados_ids, exclusivos_ids, grid_length, filename=plot_filename)


if __name__ == "__main__":
    num_clientes_arg = 15
    output_filename_arg = "instancia_generada.txt"
    do_plot = True

    if len(sys.argv) < 2:
        print("Uso: python generar_instancias.py <num_clientes> [nombre_archivo_salida] [--no-plot]")
        print(f"Ejemplo: python generar_instancias.py 10 instancia_10c.txt")
        print(f"\nUsando valores por defecto: {num_clientes_arg} clientes, archivo '{output_filename_arg}', plot activado")
    else:
        try:
            num_clientes_arg = int(sys.argv[1])
            if num_clientes_arg < 0: raise ValueError("Número de clientes no puede ser negativo.")
        except ValueError as e:
            print(f"Error: Número de clientes inválido. {e}")
            sys.exit(1)
        
        output_filename_arg = f"instancia_{num_clientes_arg}c.txt" # Default output name based on num_clients

        if len(sys.argv) > 2:
            if sys.argv[2] != "--no-plot":
                 output_filename_arg = sys.argv[2]
            else:
                do_plot = False
        
        if len(sys.argv) > 3:
            if sys.argv[3] == "--no-plot":
                do_plot = False


    grid_dim_arg = 0
    if num_clientes_arg > 0:
        base_grid_dim = math.ceil(math.sqrt(num_clientes_arg))
        if num_clientes_arg == 1:
            grid_dim_arg = 1
        elif num_clientes_arg <= 4: 
             grid_dim_arg = max(2, base_grid_dim)
        else: 
            grid_dim_arg = base_grid_dim + 1 
    
    generar_instancia(output_filename_arg, num_clientes_arg, grid_dim_arg, plot=do_plot)