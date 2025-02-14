import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import base64
import io
import csv

def expression_graph(expression, p0, root):
    '''
    This function returns the graph of the expression and its derivative
    '''
    if root is None:
        return None
    x = sp.Symbol('x')
    f_expr = expression.replace('^', '**')
    f = sp.sympify(f_expr)
    # Calculate derivative
    df = sp.diff(f, x)
    
    func = sp.lambdify(x, f, "numpy")
    dfunc = sp.lambdify(x, df, "numpy")
    
    # Generate plot range around p0 and root
    x_min = min(p0, root) - 2
    x_max = max(p0, root) + 2
    
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = func(x_vals)
    dy_vals = dfunc(x_vals)
    
    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {expression}')
    plt.plot(x_vals, dy_vals, '--', label=f"f'(x) = {df}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter(root, 0, color='red', zorder=5, label=f'Raiz en x = {root:.4f}')
    plt.grid(True)
    plt.title(f'Gráfico de f(x) y su derivada')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def newton_raphson(expression, p0, epsilon, max_iterations):
    """
    Implements Newton-Raphson method to find a root of a given function.
    Args:
        expression: String of the function
        p0: Initial guess
        epsilon: Tolerance
        max_iterations: Maximum number of iterations
    """
    try:
        x = sp.Symbol('x')
        f_expr = expression.replace('^', '**')
        f = sp.sympify(f_expr)
        # Calculate derivative
        df = sp.diff(f, x)
        
        # Convert to numpy functions
        f_num = sp.lambdify(x, f, "numpy")
        df_num = sp.lambdify(x, df, "numpy")
        
        iterations_data = []
        convergence = []
        p = p0
        
        for i in range(max_iterations):
            f_p = f_num(p)
            df_p = df_num(p)
            
            if abs(df_p) < 1e-10:  # Avoid division by zero
                return {
                    "root": None,
                    "iterations": i,
                    "error": None,
                    "convergence": convergence,
                    "success": False,
                    "message": "Derivada cercana a cero. Posible punto crítico.",
                    "table": iterations_data
                }
            
            # Newton-Raphson formula
            p_next = p - f_p/df_p
            
            iterations_data.append({
                "iteration": i + 1,
                "x_i": p,
                "f_x": f_p,
                "df_x": df_p,
                "x_next": p_next
            })
            
            convergence.append(p_next)
            
            if abs(p_next - p) < epsilon:
                return {
                    "root": p_next,
                    "iterations": i + 1,
                    "error": abs(f_num(p_next)),
                    "convergence": convergence,
                    "success": True,
                    "message": f"Raíz encontrada en {i+1} iteraciones",
                    "table": iterations_data,
                    "derivative": str(df)
                }
            
            p = p_next
        
        return {
            "root": p,
            "iterations": max_iterations,
            "error": abs(f_num(p)),
            "convergence": convergence,
            "success": False,
            "message": "Máximo de iteraciones alcanzado",
            "table": iterations_data,
            "derivative": str(df)
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

def generate_csv(data):
    """
    Generate CSV content for the iteration table.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["iteration", "x_i", "f_x", "df_x", "x_next"])
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def calculate_all_iterations(expression, p0, epsilon):
    """
    Calculates all iterations until the root is found or until the error is less than epsilon.
    """
    try:
        x = sp.Symbol('x')
        f_expr = expression.replace('^', '**')
        f = sp.sympify(f_expr)
        df = sp.diff(f, x)
        
        # Convert to numpy functions
        f_num = sp.lambdify(x, f, "numpy")
        df_num = sp.lambdify(x, df, "numpy")
        
        all_iterations_data = []
        iteration = 1
        p = p0
        
        while True:
            f_p = f_num(p)
            df_p = df_num(p)
            
            if abs(df_p) < 1e-10:  # Avoid division by zero
                return {
                    "table": [],
                    "success": False,
                    "error": "Derivada cercana a cero. Posible punto crítico."
                }
            
            # Newton-Raphson formula
            p_next = p - f_p/df_p
            
            all_iterations_data.append({
                "iteration": iteration,
                "x_i": p,
                "f_x": f_p,
                "df_x": df_p,
                "x_next": p_next
            })
            
            if abs(p_next - p) < epsilon:
                break
                
            p = p_next
            iteration += 1
            
            # Safety check to prevent infinite loops
            if iteration > 1000:
                return {
                    "table": [],
                    "success": False,
                    "error": "Excedido número máximo de iteraciones"
                }
        
        return {
            "table": all_iterations_data,
            "success": True
        }
        
    except Exception as e:
        return {
            "table": [],
            "success": False,
            "error": f"Ocurrió un error: {e}"
        }
