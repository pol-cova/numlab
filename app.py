from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from urllib.parse import unquote
import base64
from methods.bisection import expression_graph, bisection, generate_csv, calculate_all_iterations, electronic_bisection
from methods.errors import get_relative_error, get_absolute_error, get_real_error
from methods.newton_raphson import expression_graph as expression_graph_newton
from methods.newton_raphson import calculate_all_iterations as calculate_all_iterations_newton
from methods.newton_raphson import newton_raphson, generate_csv as generate_csv_newton
import json 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/biseccion")
async def bisection_page(request: Request):
    return templates.TemplateResponse("bisection.html", {"request": request})

@app.post("/biseccion")
async def calculate_bisection(request: Request):
    form_data = await request.form()
    expression = form_data["expression"]
    interval_a = float(form_data["interval_a"])
    interval_b = float(form_data["interval_b"])
    epsilon = float(form_data["epsilon"])
    iterations = int(form_data["iterations"])

    # Perform the bisection method
    result = bisection(expression, interval_a, interval_b, epsilon, iterations)

    # Graph the expression if needed
    graph_image = expression_graph(expression, interval_a, interval_b)

    # Generate CSV content if there are iterations data
    csv_data = generate_csv(result["table"]) if result["table"] else ""

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "graph_image": graph_image,
            "root": result["root"],
            "iterations": result["iterations"],
            "error": result["error"],
            "convergence": result["convergence"],
            "message": result["message"],
            "table": result["table"],
            "expression": expression,
            "interval_a": interval_a,
            "interval_b": interval_b,
            "epsilon": epsilon,
            "max_iterations": iterations,
        }
    )

@app.get("/download-csv")
async def download_csv(
    request: Request,
    expression: str = Query(...),
    a: float = Query(...),
    b: float = Query(...),
    epsilon: float = Query(...),
):
    try:
        # Decode the URL-encoded expression
        decoded_expression = unquote(expression)
        
        # Calculate iterations with decoded expression
        result = calculate_all_iterations(decoded_expression, a, b, epsilon)
        
        if not result["success"]:
            return Response(
                content=f"Error: {result.get('error', 'Unknown error')}",
                media_type="text/plain",
                status_code=400
            )
        
        csv_data = generate_csv(result["table"])
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=bisection_{decoded_expression.replace(' ', '_')}.csv"
            }
        )
    except Exception as e:
        return Response(
            content=f"Error processing request: {str(e)}",
            media_type="text/plain",
            status_code=400
        )

@app.get("/errores")
async def errores(request: Request):
    return templates.TemplateResponse("errors.html", {"request": request})

@app.post("/errores")
async def calculate_errores(request: Request):
    form_data = await request.form()
    p = float(form_data["p"])
    aprox = float(form_data["aprox"])
    # Perform the calcs
    real = get_real_error(p,aprox)
    relative =get_relative_error(p,aprox)
    absolute = get_absolute_error(p,aprox)
    porcentual = relative * 100
    return templates.TemplateResponse(
        "result-errors.html",
        {
            "request": request,
            "real": real,
            "relative": relative,
            "absolute": absolute,
            "porcentual":porcentual
        }
    )

@app.get("/newton-raphson")
async def errores(request: Request):
    return templates.TemplateResponse("newtonr.html", {"request": request})

@app.post("/newton-raphson")
async def calculate_newton_raphson(request: Request):
    form_data = await request.form()
    expression = form_data["expression"]
    x0 = float(form_data["x0"])
    epsilon = float(form_data["epsilon"])
    iterations = int(form_data["iterations"])
    # Perform the newton-raphson method
    result = newton_raphson(expression, x0, epsilon, iterations)
    # Graph the function
    graph_image = expression_graph_newton(expression, x0, result["root"]) if result["root"] is not None else None    # Generate CSV content if there are iterations data
    csv_data = generate_csv_newton(result["table"]) if result["table"] else ""
    return templates.TemplateResponse(
        "result-newton.html",
        {
            "request": request,
            "graph_image": graph_image,
            "root": result["root"],
            "iterations": result["iterations"],
            "error": result["error"],
            "convergence": result["convergence"],
            "message": result["message"],
            "table": result["table"],
            "expression": expression,
            "x0": x0,
            "epsilon": epsilon,
            "max_iterations": iterations,
        }
    )

@app.get("/download-csv-newton")
async def download_csv_newton(
    request: Request,
    expression: str = Query(...),
    x0: float = Query(...),
    epsilon: float = Query(...),
):
    try:
        decoded_expression = unquote(expression)
        result = calculate_all_iterations_newton(decoded_expression, x0, epsilon)  # Use reasonable max_iterations
        
        if not result["success"]:
            return Response(
                content=f"Error: {result.get('error', 'Unknown error')}",
                media_type="text/plain",
                status_code=400
            )
        
        csv_data = generate_csv_newton(result["table"])
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=newton_{decoded_expression.replace(' ', '_')}.csv"
            }
        )
    except Exception as e:
        return Response(
            content=f"Error processing request: {str(e)}",
            media_type="text/plain",
            status_code=400
        )

@app.get("/secante")
async def secantes(request: Request):
    return templates.TemplateResponse("secante.html", {"request": request})

# Power electronics
@app.get("/schockley")
async def schockley(request: Request):
    return templates.TemplateResponse("schockley.html", {"request": request})

@app.post("/schockley")
async def calculate_schockley(request: Request):
    form_data = await request.form()
    try:
        # Get form data and convert to appropriate types
        vd1 = float(form_data["vd1"])
        id1 = float(form_data["id1"])
        vd2 = float(form_data["vd2"])
        id2 = float(form_data["id2"])
        epsilon = float(form_data["epsilon"])
        iterations = int(form_data["iterations"])
        temp = float(form_data["temp"])

        # Perform the electronic bisection method
        result = electronic_bisection(vd1, id1, vd2, id2, epsilon, iterations, temp)

        return templates.TemplateResponse(
            "result-schockley.html",
            {
                "request": request,
                "Is": result["Is"],
                "n": result["n"],
                "iterations": result["iterations"],
                "error": result["error"],
                "message": result["message"],
                "table": result["table"],
                "success": result["success"],
                "is_formatted": result["is_formatted"],
                "n_formatted": result["n_formatted"],
                "vd1": vd1,
                "id1": id1,
                "vd2": vd2,
                "id2": id2,
                "temp": temp
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "result-schockley.html",
            {
                "request": request,
                "success": False,
                "message": f"Error en el cálculo: {str(e)}",
                "table": []
            }
        )


# Utils routes for development


# Default in construction route for new future pages
@app.get("/construction")
async def errores(request: Request):
    return templates.TemplateResponse("construction.html", {"request": request})

@app.exception_handler(404)
async def custom_404_handler(request, __):
    return templates.TemplateResponse("404.html", {"request": request})