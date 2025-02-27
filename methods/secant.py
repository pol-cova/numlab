import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import base64
import io
import csv

def secant_graph(expression, x0, x1, root=None):
    """
    Generates a graph of the function with secant lines.

    Args:
        expression (str): Function expression as a string.
        x0 (float): First initial guess.
        x1 (float): Second initial guess.
        root (float, optional): Calculated root.

    Returns:
        str: Base64 encoded image of the graph.
    """
    x = sp.Symbol('x')
    f_expr = expression.replace('^', '**')
    f = sp.sympify(f_expr)
    func = sp.lambdify(x, f, "numpy")
    
    # Define plot range
    x_min, x_max = min(x0, x1) - 2, max(x0, x1) + 2
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = func(x_vals)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {expression}')
    
    # Plot secant line
    y0, y1 = func(x0), func(x1)
    plt.plot([x0, x1], [y0, y1], '--r', label='Secante')

    # Mark root if available
    if root is not None:
        plt.scatter(root, 0, color='green', zorder=5, label=f'Raíz en x = {root:.4f}')
    
    # Mark points
    plt.scatter([x0, x1], [y0, y1], color='red', zorder=5)
    
    # Axes and grid
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title('Gráfico de f(x) y línea secante')
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

def secant_method(expression, x0, x1, epsilon=1e-6, max_iterations=100):
    """
    Implements the Secant method to approximate the root of a function.

    Args:
        expression (str): Function as a string.
        x0 (float): First initial approximation.
        x1 (float): Second initial approximation.
        epsilon (float): Error tolerance.
        max_iterations (int): Maximum iterations.

    Returns:
        dict: Contains the root, error, iteration count, convergence list, and success status.
    """
    try:
        x = sp.Symbol('x')
        f_expr = expression.replace('^', '**')
        f = sp.sympify(f_expr)
        f_num = sp.lambdify(x, f, "numpy")
        
        iterations_data = []
        convergence = []
        
        for i in range(1, max_iterations + 1):
            f_x0, f_x1 = f_num(x0), f_num(x1)

            if abs(f_x1 - f_x0) < 1e-10:
                return {
                    "root": None,
                    "iterations": i,
                    "error": None,
                    "convergence": convergence,
                    "success": False,
                    "message": "División por cero detectada.",
                    "table": iterations_data
                }
            
            # Compute next approximation
            x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            abs_error = abs(x_next - x1)
            f_next = f_num(x_next)

            iterations_data.append({
                "iteration": i,
                "Pn": x_next,
                "f(Pn)": f_next,
                "absolute error": abs_error
            })
            
            convergence.append(x_next)

            # Stopping criteria
            if abs_error < epsilon or abs(f_next) < epsilon:
                return {
                    "root": x_next,
                    "iterations": i,
                    "error": abs(f_next),
                    "convergence": convergence,
                    "success": True,
                    "message": f"Raíz encontrada en {i} iteraciones",
                    "table": iterations_data
                }
            
            x0, x1 = x1, x_next
        
        return {
            "root": None,
            "iterations": max_iterations,
            "error": abs(f_num(x1)),
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

def generate_csv_secant(data, filename="secant_results.csv"):
    """
    Generates a CSV file from the iteration table.

    Args:
        data (list): List of dictionaries containing iteration data.
        filename (str): Output filename.

    Returns:
        str: CSV content.
    """
    output = io.StringIO()
    fieldnames = ["iteration", "Pn", "f(Pn)", "absolute error"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()
