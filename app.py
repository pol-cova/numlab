from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from urllib.parse import unquote

from methods.bisection import expression_graph, bisection, generate_csv, calculate_all_iterations

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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