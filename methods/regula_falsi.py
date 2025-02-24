import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import base64
import io
import csv

def regula_falsi_graph(expression, a, b, root=None):
    """
    Generates a graph of the function with the false position line.

    Args:
        expression (str): Function expression as a string.
        a (float): Left endpoint of interval.
        b (float): Right endpoint of interval.
        root (float, optional): Calculated root.

    Returns:
        str: Base64 encoded image of the graph.
    """
    x = sp.Symbol('x')
    f_expr = expression.replace('^', '**')
    f = sp.sympify(f_expr)
    func = sp.lambdify(x, f, "numpy")
    
    # Define plot range
    x_min, x_max = min(a, b) - 2, max(a, b) + 2
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = func(x_vals)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {expression}')
    
    # Plot false position line
    y_a, y_b = func(a), func(b)
    plt.plot([a, b], [y_a, y_b], '--r', label='Línea de Falsa Posición')

    # Mark root if available
    if root is not None:
        plt.scatter(root, 0, color='green', zorder=5, label=f'Raíz en x = {root:.4f}')
    
    # Mark interval points
    plt.scatter([a, b], [y_a, y_b], color='red', zorder=5)
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title('Gráfico de f(x) y Método de Falsa Posición')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Convert to Base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def regula_falsi_method(expression, a, b, epsilon=1e-6, max_iterations=100):
    """
    Implements the Regula Falsi method to find a root.

    Args:
        expression (str): Function as a string.
        a (float): Left endpoint of interval.
        b (float): Right endpoint of interval.
        epsilon (float): Error tolerance.
        max_iterations (int): Maximum iterations.

    Returns:
        dict: Contains the root, error, iteration count, and convergence data.
    """
    try:
        x = sp.Symbol('x')
        f_expr = expression.replace('^', '**')
        f = sp.sympify(f_expr)
        f_num = sp.lambdify(x, f, "numpy")
        
        f_a = f_num(a)
        f_b = f_num(b)
        
        # Check if the interval is valid
        if f_a * f_b >= 0:
            return {
                "root": None,
                "iterations": 0,
                "error": None,
                "convergence": [],
                "success": False,
                "message": "El intervalo no es válido: f(a) y f(b) deben tener signos opuestos",
                "table": []
            }
        
        iterations_data = []
        convergence = []
        prev_c = None
        
        for i in range(1, max_iterations + 1):
            # Calculate the false position point
            c = b - (f_b * (b - a)) / (f_b - f_a)
            f_c = f_num(c)
            
            # Calculate absolute error if possible
            abs_error = abs(c - prev_c) if prev_c is not None else None
            
            iterations_data.append({
                "iteration": i,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": f_c,
                "absolute error": abs_error
            })
            
            convergence.append(c)
            
            # Check for convergence
            if abs_error is not None and abs_error < epsilon:
                return {
                    "root": c,
                    "iterations": i,
                    "error": abs(f_c),
                    "convergence": convergence,
                    "success": True,
                    "message": f"Raíz encontrada en {i} iteraciones",
                    "table": iterations_data
                }
            
            # Update interval
            if f_c * f_b < 0:
                a = a
                f_a = f_a
            else:
                a = c
                f_a = f_c
            
            prev_c = c
        
        return {
            "root": c,
            "iterations": max_iterations,
            "error": abs(f_c),
            "convergence": convergence,
            "success": False,
            "message": "Máximo de iteraciones alcanzado",
            "table": iterations_data
        }
        
    except Exception as e:
        return {
            "root": None,
            "iterations": 0,
            "error": None,
            "convergence": [],
            "success": False,
            "message": f"Ocurrió un error: {e}",
            "table": []
        }

def generate_csv_regula_falsi(data, filename="regula_falsi_results.csv"):
    """
    Generates a CSV file from the iteration table.

    Args:
        data (list): List of dictionaries containing iteration data.
        filename (str): Output filename.

    Returns:
        str: CSV content.
    """
    output = io.StringIO()
    fieldnames = ["iteration", "a", "b", "c", "f(c)", "absolute error"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()