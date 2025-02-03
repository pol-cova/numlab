import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.optimize as opt
import base64
import io
import csv

def expression_graph(expression, interval_a, interval_b):
    '''
    This function returns the graph of the expression encoded as a base64 image, 
    along with the point where the graph intersects the x-axis.
    '''
    x = sp.Symbol('x')
    f_expr = expression.replace('^', '**')
    f = sp.sympify(f_expr)
    func = sp.lambdify(x, f, "numpy")
    
    # Extend the plot range beyond the given interval
    x_min = min(interval_a, interval_b) - 5  # Extend left of the interval
    x_max = max(interval_a, interval_b) + 5  # Extend right of the interval
    
    # Generate x values for the plot
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = func(x_vals)
    
    # Find the root (x-intercept) using fsolve, starting from the middle of the interval
    root = opt.fsolve(func, (interval_a + interval_b) / 2)[0]
    
    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {expression}')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter(root, 0, color='red', zorder=5, label=f'Raiz en x = {root:.4f}')
    plt.grid(True)
    plt.title(f'Grafico de f(x): {expression}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    # Save to base64 for embedding in HTML
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def bisection(expression, interval_a, interval_b, epsilon, iterations):
    """
    Implements the bisection method to find a root of a given function within a specified interval.
    Returns:
        dict: Dictionary with root, convergence details, success status, and iteration table data
    """
    try:
        # Symbolic parsing of the expression
        x = sp.symbols('x')
        f_expr = sp.sympify(expression)

        # Evaluate the function
        f = sp.lambdify(x, f_expr, "numpy")

        # Check if the function has opposite signs at a and b
        if f(interval_a) * f(interval_b) >= 0:
            return {
                "root": None,
                "iterations": 0,
                "error": None,
                "convergence": [],
                "success": False,
                "message": "No existe raiz en ese intervalo.",
                "table": []
            }

        a, b = interval_a, interval_b
        iterations_data = []
        convergence = []

        for i in range(iterations):
            # Midpoint p
            p = (a + b) / 2
            f_a = f(a)
            f_p = f(p)

            # Add to iteration table
            iterations_data.append({
                "iteration": i + 1,
                "a": a,
                "b": b,
                "p": p,
                "f_a": f_a,
                "f_p": f_p
            })

            # Store the current approximation
            convergence.append(p)

            # Bisection step
            if abs(f_p) < epsilon:
                break
            if f_a * f_p < 0:
                b = p
            else:
                a = p

        # Ensure correct success check using the final values of f_a and f_b
        f_b = f(b) 
        root = convergence[-1] if convergence else None
        error = abs(f(root)) if root is not None else None
        success = f_a * f_b < 0
        message = f"Raíz encontrada en {i+1} iteraciones" if success else "No se pudo encontrar la raíz"

        return {
            "root": root,
            "iterations": i + 1,
            "error": error,
            "convergence": convergence,
            "success": success,
            "message": message,
            "table": iterations_data 
        }

    except Exception as e:
        return {
            "root": None,
            "iterations": 0,
            "error": None,
            "convergence": [],
            "success": False,
            "message": f"Ocurrio un error: {e}",
            "table": []
        }

def calculate_all_iterations(expression, interval_a, interval_b, epsilon):
    """
    Calculates all iterations until the root is found or until the error is less than epsilon.
    """
    try:
        # Symbolic parsing of the expression
        x = sp.symbols('x')
        f_expr = expression.replace('^', '**')
        f = sp.sympify(f_expr)
        f = sp.lambdify(x, f, "numpy")

        a, b = interval_a, interval_b
        all_iterations_data = []
        iteration = 1

        # Check if the function has opposite signs at a and b
        if f(interval_a) * f(interval_b) >= 0:
            return {
                "table": [],
                "success": False,
                "error": "La función tiene el mismo signo en los extremos del intervalo"
            }

        while True:
            p = (a + b) / 2
            f_a = f(a)
            f_p = f(p)

            all_iterations_data.append({
                "iteration": iteration,  
                "a": a,
                "b": b,
                "p": p,
                "f_a": f_a,
                "f_p": f_p
            })

            if abs(f_p) < epsilon:
                break

            if f_a * f_p < 0:
                b = p
            else:
                a = p
            
            iteration += 1

        return {
            "table": all_iterations_data,
            "success": True
        }

    except Exception as e:
        return {
            "table": [],
            "success": False,
            "error": f"Ocurrio un error: {e}"
        }

def generate_csv(data):
    """
    Generate CSV content for the iteration table.
    
    Parameters:
        data (list): List of dictionaries with iteration data (a, b, p, f(a), f(p))
    
    Returns:
        str: CSV formatted data
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["iteration", "a", "b", "p", "f_a", "f_p"])
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()
