# NumLab

NumLab is a web application designed to help students and enthusiasts learn and apply numerical methods interactively. Built with **FastAPI** on the backend and **HTMX** on the frontend, it provides a dynamic environment for solving mathematical problems and implementing numerical algorithms.

## Why NumLab?

During our studies, my classmates and I found it challenging to understand and apply numerical methods effectively. Traditional learning methods often lack interactivity, making it harder to grasp concepts deeply. **NumLab** was created to bridge this gap by offering an intuitive platform where users can experiment with numerical methods in real-time. Whether you're solving equations, performing interpolations, or working on differential equations, **NumLab** aims to make numerical methods more accessible and engaging.

## Features

- 🧮 **Interactive Numerical Methods** – Solve problems using various numerical techniques.
- ⚡ **FastAPI Backend** – Ensuring high performance and scalability.
- 🔥 **HTMX-Based Frontend** – Providing a smooth and reactive UI.
- 📊 **Visualization Support** – Generate plots and graphs for better understanding.
- 🛠 **Extensible** – Easily add new numerical methods and expand functionality.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** HTMX, TailwindCSS
- **Database:** SQLite (or other relational databases)
- **Other Tools:** NumPy, SciPy, Matplotlib (for numerical computations and visualizations)

## Getting Started

### Prerequisites

- Python 3.8+
- Pipenv or virtualenv (recommended for dependency management)

### Installation

```sh
# Clone the repository
git clone https://github.com/yourusername/numlab.git
cd numlab

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Then, open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000) to start using NumLab!

## Contribution

Contributions are welcome! If you’d like to improve **NumLab**, follow these steps:

1. **Fork** the repository.
2. **Clone** your fork: `git clone https://github.com/yourusername/numlab.git`
3. **Create a new branch** for your feature: `git checkout -b feature-name`
4. **Make your changes** and commit: `git commit -m "Add new feature"`
5. **Push** to your fork: `git push origin feature-name`
6. **Create a Pull Request** – we’ll review it as soon as possible!

### Code Guidelines

- Follow **PEP 8** for Python code.
- Use meaningful commit messages.
- Keep the code clean and documented.

## License

NumLab is released under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

💡 **Let's make numerical methods more accessible together!** 🚀
