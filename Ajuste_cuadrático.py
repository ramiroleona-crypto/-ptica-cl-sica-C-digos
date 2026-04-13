import numpy as np
import matplotlib.pyplot as plt

# angulos registrados experimentalmente
alpha_deg = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
beta_deg  = np.array([6.0, 13.0, 19.0, 25.0, 31.0])

# incertidumbre angular
incertidumbre_angular_deg = 0.1

# conversión a radianes
alpha = np.deg2rad(alpha_deg)
beta  = np.deg2rad(beta_deg)
incertidumbre = np.deg2rad(incertidumbre_angular_deg)

# datos necesarios
sin_alpha = np.sin(alpha)
cos_alpha = np.cos(alpha)
sin_beta  = np.sin(beta)
cos_beta  = np.cos(beta)

# propagacion de errores
sigma_sin_alpha = np.abs(cos_alpha) * incertidumbre
sigma_sin_beta  = np.abs(cos_beta)  * incertidumbre

# --- Ajuste ---------------------------------------------------------------------------------------------------

# m0 ignorando el error en sin_beta
w = 1 / sigma_sin_alpha**2
m0 = np.sum(w * sin_alpha * sin_beta) / np.sum(w * sin_beta**2)

# sigma efectivo
sigma_eff = np.sqrt(sigma_sin_alpha**2 + (m0**2)*(sigma_sin_beta**2))
W = 1 / sigma_eff**2

# pendiente final
m = np.sum(W * sin_alpha * sin_beta) / np.sum(W * sin_beta**2)

# error en m
sigma_m = np.sqrt(1 / np.sum(W * sin_beta**2))

# chi cuadrado reducido
chi2 = np.sum(((sin_alpha - m*sin_beta)**2) / sigma_eff**2)
chi2_red = chi2 / (len(sin_beta) - 1)

# correccion
sigma_m *= np.sqrt(chi2_red)
print(sin_alpha, sigma_sin_alpha)
print(sin_beta, sigma_sin_beta)
print(f"n = {m:.3f} ± {sigma_m:.3f}")

# --- Grafica ------------------------------------------------------

plt.figure(figsize=(6,5))

factor = 20

plt.errorbar(sin_beta, sin_alpha,
             yerr=factor*sigma_sin_alpha,
             fmt='o', capsize=4, label='Datos (errores escalados)')

# etiquetas con 3 cifras significativas
for xi, yi in zip(sin_beta, sin_alpha):
    plt.text(xi, yi + 0.01, f'({xi:.3g}, {yi:.3g})', fontsize=8)

# recta ajustada
x_fit = np.linspace(min(sin_beta), max(sin_beta), 100)
y_fit = m * x_fit
plt.plot(x_fit, y_fit, label=f'Ajuste: n = {m:.4g} ± {sigma_m:.2g}')

# etiquetas
plt.xlabel('sin(beta)')
plt.ylabel('sin(alpha)')
plt.legend()
plt.grid()

plt.show()