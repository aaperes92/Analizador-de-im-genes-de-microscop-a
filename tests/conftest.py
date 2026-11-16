"""Configuración común de pytest para todo el proyecto."""

import matplotlib

# Backend sin ventana: plt.show() no bloquea durante los tests
matplotlib.use("Agg")
