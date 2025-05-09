# Графическая схема времени Любона с резонансами, эффектом H и космологией
# Оптимизировано для Binder и мобильных устройств

# Установка зависимостей
!pip install --no-cache-dir --force-reinstall numpy==1.26.4 matplotlib==3.9.2 mpmath==1.3.0 scipy==1.14.1 plotly==5.24.1 sympy==1.13.3

# Проверка установки
try:
    import numpy, mpmath, plotly, scipy, sympy
    print(f"NumPy: {numpy.__version__}, mpmath: {mpmath.__version__}, Plotly: {plotly.__version__}, SciPy: {scipy.__version__}, SymPy: {sympy.__version__}")
except ImportError as e:
    print(f"Ошибка установки: {e}")

# Импорт библиотек
import numpy as np
import plotly.graph_objects as go
from mpmath import mp, chi
from sympy import zeta as sympy_zeta
from functools import lru_cache
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# График 1: Любо́н (Lub)

# Настройки mpmath
mp.dps = 30

# Параметры
T = 50
t = np.linspace(-T, T, 300)
gamma_n = [14.1347, 21.0220, 25.0108]
k = 1.0
epsilon = 1e-8
omega = 1e12
hbar = 6.582e-16
dt_avg = 2 * T / len(t)

# Кэширование zeta и chi
@lru_cache(maxsize=1000)
def cached_zeta(s_real, s_imag):
    try:
        return complex(sympy_zeta(complex(s_real, s_imag)))
    except:
        return 0.0

@lru_cache(maxsize=1000)
def cached_chi(s_real, s_imag):
    try:
        return complex(chi(complex(s_real, s_imag)))
    except:
        return 0.0

# Модуль |zeta(1/2 + it)|^2
def zeta_squared(u):
    z = cached_zeta(0.5, u)
    result = z * z.conjugate()
    return float(result.real) if not np.isnan(result) and not np.isinf(result) else 0.0

# Функция psi
def psi(s, u, epsilon):
    denom = (s - 0.5 - 1j*u)**2 * (1 - s - 0.5 - 1j*u)**2 + epsilon**2
    result = 1 / denom
    zeta_s = cached_zeta(s.real, s.imag)
    chi_s = cached_chi(s.real, s.imag)
    chi_zeta = abs(chi_s * zeta_s)**2
    if chi_zeta > 1e10:
        chi_zeta = 1e10
    result *= chi_zeta
    return result if not np.isnan(result) and not np.isinf(result) else 0.0

# Вычисление ядра K_sym
def compute_kernel(epsilon):
    N = len(t)
    K = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            s = 0.5 + 1j * t[i]
            u = t[j]
            psi_val = psi(s, u, epsilon)
            zeta_val = zeta_squared(u)
            K[i, j] = psi_val * zeta_val
    K = (K + K.conj().T) / 2
    return K

# Потенциал
V_eff = -k * np.array([zeta_squared(ti) for ti in t])

# Вычисление собственных функций
print("Шаг 1: Вычисление ядра и собственных функций")
K_sym = compute_kernel(epsilon) * dt_avg
eigvals, eigvecs = np.linalg.eigh(K_sym)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Выбор трёх собственных функций
selected_indices = [np.argmin(np.abs(eigvals - gamma)) for gamma in gamma_n]
fn_data = [np.abs(eigvecs[:, idx])**2 for idx in selected_indices]

# Интерполяция
fn_interps = [interp1d(t, fn, kind='cubic') for fn in fn_data]

# Плотности вероятности с фазой
psi_data = []
for i, gamma in enumerate(gamma_n):
    frames = []
    for time in np.linspace(0, 1e-12, 20):
        phase = omega * gamma * time
        psi = fn_interps[i](t) * np.exp(1j * phase)
        psi_squared = np.abs(psi)**2
        frames.append(psi_squared)
    psi_data.append(frames)

# Объяснение графика 1
print("\nОбъяснение графика 1:")
print("График показывает анимированную 3D-визуализацию Любо́на. Синие, зелёные и фиолетовые линии отображают плотности вероятности |lub_n(t)|^2 для резонансов γ_1=14.1347, γ_2=21.0220, γ_3=25.0108, связанных с нулями дзета-функции Римана. Красные маркеры обозначают резонансы (γ_n), а синие — эффект оператора H, стабилизирующего волновые функции. Красная пунктирная линия и полупрозрачная красная поверхность представляют потенциал V_eff(t) = -|ζ(1/2 + it)|^2. Слайдеры позволяют переключать резонансы, а кнопки управляют анимацией, показывая динамику фазы.")

# Создание графика
fig1 = go.Figure()

# Волновые функции
colors = ['blue', 'green', 'purple']
for i, gamma in enumerate(gamma_n):
    fig1.add_trace(go.Scatter3d(
        x=t,
        y=psi_data[i][0],
        z=V_eff,
        mode='lines',
        line=dict(width=5, color=colors[i]),
        name=f'|lub_{i+1}(t)|^2 (γ_{i+1} = {gamma})',
        visible=(i == 0)
    ))

# Потенциал
fig1.add_trace(go.Scatter3d(
    x=t,
    y=np.zeros_like(t),
    z=V_eff,
    mode='lines',
    line=dict(width=4, color='red', dash='dash'),
    name='V_eff(t)'
))

# Поверхностный график
X, Y = np.meshgrid(t, np.linspace(0, max(psi_data[0][0]) * 1.1, 20))
Z = np.tile(V_eff, (20, 1))
fig1.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z,
    opacity=0.3,
    colorscale='Reds',
    showscale=False,
    name='Потенциал (поверхность)'
))

# Резонансы и эффект H
resonance_indices = [np.argmin(np.abs(t - gamma)) for gamma in gamma_n]
fig1.add_trace(go.Scatter3d(
    x=t[resonance_indices],
    y=[max(psi_data[0][0]) * 0.5] * len(gamma_n),
    z=V_eff[resonance_indices],
    mode='markers+text',
    marker=dict(size=8, color='red', symbol='circle'),
    text=['Резонанс'] * len(gamma_n),
    textposition='top center',
    textfont=dict(size=12, color='black', family='Arial', weight='bold'),
    name='Резонансы (γ_n)'
))
fig1.add_trace(go.Scatter3d(
    x=t[resonance_indices],
    y=[max(psi_data[0][0]) * 0.5] * len(gamma_n),
    z=V_eff[resonance_indices],
    mode='markers+text',
    marker=dict(size=10, color='blue', symbol='circle'),
    text=['Эффект H'] * len(gamma_n),
    textposition='bottom center',
    textfont=dict(size=12, color='black', family='Arial', weight='bold'),
    name='Эффект H'
))

# Анимация
frames = []
for f in range(20):
    frame_data = []
    for i, gamma in enumerate(gamma_n):
        frame_data.append(go.Scatter3d(
            x=t,
            y=psi_data[i][f],
            z=V_eff,
            mode='lines',
            line=dict(width=5, color=colors[i]),
            name=f'|lub_{i+1}(t)|^2',
            visible=(i == 0)
        ))
    frame_data.append(fig1.data[len(gamma_n)])
    frame_data.append(fig1.data[len(gamma_n) + 1])
    frame_data.append(fig1.data[len(gamma_n) + 2])
    frame_data.append(fig1.data[len(gamma_n) + 3])
    frames.append(go.Frame(data=frame_data, name=f'frame{f}'))

fig1.frames = frames

# Слайдеры
sliders = [
    dict(
        steps=[
            dict(
                method='update',
                args=[{'visible': [k == i for k in range(len(gamma_n))] + [True, True, True, True]},
                      {'title': f'Любо́н: |lub_{i+1}(t)|^2 (γ_{i+1} = {gamma_n[i]})'}],
                label=f'γ_{i+1}'
            ) for i in range(len(gamma_n))
        ],
        active=0,
        currentvalue={'prefix': 'Выбор нуля: '},
        pad={'t': 50}
    )
]

# Кнопки анимации
fig1.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='Играть',
                    method='animate',
                    args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                ),
                dict(
                    label='Пауза',
                    method='animate',
                    args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                )
            ],
            pad={'r': 10, 't': 10}
        )
    ],
    sliders=sliders,
    title=dict(
        text='Анимированная 3D-визуализация Любо́на (Lub)',
        x=0.0,
        xanchor='left',
        font=dict(size=16, weight='bold', family='Arial')
    ),
    scene=dict(
        xaxis_title='t',
        yaxis_title='|lub(t)|^2',
        zaxis_title='V_eff(t)',
        xaxis=dict(range=[-T, T]),
        yaxis=dict(range=[0, max(psi_data[0][0]) * 1.1]),
        zaxis=dict(range=[min(V_eff) * 1.1, max(V_eff) * 1.1])
    ),
    showlegend=True,
    width=500,
    height=600,
    margin=dict(l=50, r=20, t=50, b=20),
    scene_aspectmode='manual',
    scene_aspectratio=dict(x=1, y=1, z=0.5),
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
    legend=dict(
        x=0.0,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    )
)

print("Отображение графика 1: Анимированная 3D-визуализация Любо́на")
fig1.show()

print("* График сохранён как lubon_visualization.html")
fig1.write_html("lubon_visualization.html")

# Функция регуляризации для эффекта H
def regularize_factor(t):
    return np.exp(-0.01 * np.abs(t) * np.log(np.maximum(np.abs(t) + 1, 2)))

# График 2: Время как спираль Любона

# Параметры
T = 20
t = np.linspace(-T, T, 500)
gamma_n = [14.1347, 21.0220, 25.0108, 30.4249, 32.9351]
omega = 1e3
c_n = [6.0, 2.5, 2.0, 1.5, 1.2]
scale_factor = 10000.0
axis_range = 30
time_range = 20

# Простое ядро для psi_n(t)
print("\nШаг 2: Вычисление базовых функций psi_n(t)")
def psi_n_simple(t, gamma):
    s = complex(0.5, t)
    z = cached_zeta(0.5, t)
    return np.exp(-1j * omega * gamma * t) * float(abs(z))

# Поле Любона с эффектом H
print("Шаг 3: Поле phi(t) с H")
def phi_t(t_val):
    phi = 0
    for i, gamma in enumerate(gamma_n):
        phi += c_n[i] * psi_n_simple(t_val, gamma)
    return phi * scale_factor * regularize_factor(t_val)

# Вычисление phi_vals
phi_vals = np.array([phi_t(ti) for ti in t])
print("Диагностика phi_vals (график 2):")
print(f"Мин. Re(phi): {np.min(np.real(phi_vals)):.2f}, Макс. Re(phi): {np.max(np.real(phi_vals)):.2f}")
print(f"Мин. Im(phi): {np.min(np.imag(phi_vals)):.2f}, Макс. Im(phi): {np.max(np.imag(phi_vals)):.2f}")
print(f"NaN в phi_vals: {np.any(np.isnan(phi_vals))}")
print(f"Inf в phi_vals: {np.any(np.isinf(phi_vals))}")
print(f"Пример phi_vals[0]: {phi_vals[0]}")
print(f"Средняя амплитуда: {np.mean(np.abs(phi_vals)):.2f}")

# Топологические заряды
print("Шаг 4: Топологические заряды")
charges = []
for n, gamma in enumerate(gamma_n):
    psi_vals = np.array([psi_n_simple(ti, gamma) for ti in t])
    phase = np.unwrap(np.angle(psi_vals))
    grad_phase = np.gradient(phase, t)
    q_n = np.trapz(grad_phase, t) / (2 * np.pi)
    charges.append(q_n)
    print(f"q_{n+1} (γ={gamma:.1f}) = {q_n:.2f}")

# Резонансы и эффект H
resonance_times = [gamma_n[0], gamma_n[1]]
resonance_indices = [np.argmin(np.abs(t - gamma)) for gamma in resonance_times]

# Визуализация
print("Шаг 5: Графическая схема времени (график 2)")
colors = np.linspace(0, 1, len(t))
x_range = [np.min(np.real(phi_vals)), np.max(np.real(phi_vals))]
y_range = [np.min(np.imag(phi_vals)), np.max(np.imag(phi_vals))]
x_range = [x_range[0] * 1.5, x_range[1] * 1.5] if x_range[0] != x_range[1] else [-1, 1]
y_range = [y_range[0] * 1.5, y_range[1] * 1.5] if y_range[0] != y_range[1] else [-1, 1]

fig2 = go.Figure(data=[
    go.Scatter3d(x=np.real(phi_vals), y=np.imag(phi_vals), z=t, mode='lines', line=dict(color=colors, colorscale='Viridis', width=2), name='Спираль времени', legendgroup='spiral', showlegend=False),
    go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color=colors[0], colorscale='Viridis', width=2), name='Спираль времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color=colors[len(t)//2], colorscale='Viridis', width=2), name='Спираль времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color=colors[-1], colorscale='Viridis', width=2), name='Спираль времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals[::100]), y=np.imag(phi_vals[::100]), z=t[::100], mode='markers', marker=dict(size=8, color='red', symbol='circle'), name='Резонансы (γ_n)', legendgroup='resonances', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals[resonance_indices]), y=np.imag(phi_vals[resonance_indices]), z=t[resonance_indices], mode='markers+text', marker=dict(size=10, color='blue', symbol='circle'), text=['Эффект H'] * len(resonance_indices), textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Эффект H', legendgroup='h_effect', showlegend=True),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_range, axis_range], mode='lines', line=dict(color='black', width=5, dash='dash'), name='Ось смысла', legendgroup='axis', showlegend=True),
    go.Scatter3d(x=[-axis_range, axis_range], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5, dash='dash'), name='Ось смысла', legendgroup='axis', showlegend=False),
    go.Scatter3d(x=[0, 0], y=[-axis_range, axis_range], z=[0, 0], mode='lines', line=dict(color='black', width=5, dash='dash'), name='Ось смысла', legendgroup='axis', showlegend=False),
    go.Scatter3d(x=[0], y=[0], z=[axis_range + 2], mode='text', text=['Ось смысла'], textfont=dict(size=14, color='black', family='Arial', weight='bold'), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[axis_range + 2, axis_range + 0.5], mode='lines', line=dict(color='black', width=2), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-time_range, time_range], mode='lines', line=dict(color='purple', width=5), name='Ось времени', legendgroup='time_axis', showlegend=True),
    go.Scatter3d(x=[0], y=[0], z=[time_range + 2], mode='text', text=['Ось времени'], textfont=dict(size=14, color='black', family='Arial', weight='bold'), showlegend=False)
])

fig2.update_layout(
    scene=dict(
        xaxis_title='Re(φ(t))',
        yaxis_title='Im(φ(t))',
        zaxis_title='Время (с)',
        aspectmode='cube',
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        zaxis=dict(range=[-axis_range - 3, axis_range + 3])
    ),
    title=dict(text='Время как спираль Любона', x=0.5, font=dict(size=16, weight='bold', family='Arial')),
    showlegend=True,
    coloraxis_showscale=False,
    width=500,
    height=600,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        x=0.0,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    ),
    scene_dragmode='orbit',
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25))
)

print("\nОписание графика 2:")
print("Сложная 3D-спираль поля Любона, сглаженная оператором H. Красные точки — резонансы (γ_n), синие маркеры с подписью 'Эффект H' показывают влияние H на γ_1, γ_2. Чёрная ось смысла и фиолетовая ось времени (±20 с) задают структуру. Цвет (Viridis) отражает течение времени.")

print("Отображение графика 2: Время как спираль Любона")
fig2.show()

# График 3: Динамичная 3D-траектория

print("\nШаг 6: График 3 (Динамичная 3D-траектория)")

t3 = np.linspace(-60, 60, 1000)
phi_vals3 = np.exp(1j * t3) * np.sin(t3) * regularize_factor(t3)
axis_range3 = 30
time_range3 = 60
resonance_indices3 = [np.argmin(np.abs(t3 - gamma)) for gamma in resonance_times]

fig3 = go.Figure(data=[
    go.Scatter3d(x=np.real(phi_vals3), y=np.imag(phi_vals3), z=t3, mode='lines', line=dict(color='green', width=2), name='Спираль времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals3[::100]), y=np.imag(phi_vals3[::100]), z=t3[::100], mode='markers', marker=dict(size=8, color='red', symbol='circle'), name='Резонансы (γ_n)', legendgroup='resonances', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals3[resonance_indices3]), y=np.imag(phi_vals3[resonance_indices3]), z=t3[resonance_indices3], mode='markers+text', marker=dict(size=10, color='blue', symbol='circle'), text=['Эффект H'] * len(resonance_indices3), textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Эффект H', legendgroup='h_effect', showlegend=True),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_range3, axis_range3], mode='lines', line=dict(color='black', width=5, dash='dot'), name='Ось смысла', legendgroup='axis', showlegend=True),
    go.Scatter3d(x=[-axis_range3, axis_range3], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5, dash='dot'), name='Ось смысла', legendgroup='axis', showlegend=False),
    go.Scatter3d(x=[0, 0], y=[-axis_range3, axis_range3], z=[0, 0], mode='lines', line=dict(color='black', width=5, dash='dot'), name='Ось смысла', legendgroup='axis', showlegend=False),
    go.Scatter3d(x=[0], y=[0], z=[axis_range3 + 2], mode='text', text=['Ось смысла'], textfont=dict(size=14, color='black', family='Arial', weight='bold'), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[axis_range3 + 2, axis_range3 + 0.5], mode='lines', line=dict(color='black', width=2), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-time_range3, time_range3], mode='lines', line=dict(color='purple', width=5), name='Ось времени', legendgroup='time_axis', showlegend=True),
    go.Scatter3d(x=[0], y=[0], z=[time_range3 + 2], mode='text', text=['Ось времени'], textfont=dict(size=14, color='purple', family='Arial', weight='bold'), showlegend=False)
])

fig3.update_layout(
    scene=dict(
        xaxis_title='Re(φ(t))',
        yaxis_title='Im(φ(t))',
        zaxis_title='Время (с)',
        aspectmode='cube',
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-0.5, 1.5]),
        zaxis=dict(range=[-time_range3 - 3, time_range3 + 3])
    ),
    title=dict(text='Динамичная 3D-траектория', x=0.5, font=dict(size=16, weight='bold', family='Arial')),
    showlegend=True,
    width=500,
    height=600,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        x=0.0,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    ),
    scene_dragmode='orbit',
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25))
)

print("\nОписание графика 3:")
print("Зелёная спираль, сглаженная H, показывает циклическое движение (±60 с). Красные точки — резонансы (γ_n), синие маркеры с 'Эффект H' — влияние H на γ_1, γ_2. Чёрная ось смысла и фиолетовая ось времени задают структуру.")

print("Отображение графика 3: Динамичная 3D-траектория")
fig3.show()

# График 4: Спираль осознания

print("\nШаг 7: График 4 (Спираль осознания)")

t4 = np.linspace(-60, 60, 1000)
phi_vals4 = np.exp(1j * t4) * np.sin(t4) * regularize_factor(t4)
axis_range4 = 30
time_range4 = 60
collapse_idx = np.argmin(np.abs(t4))
resonance_indices4 = [np.argmin(np.abs(t4 - gamma)) for gamma in resonance_times]

fig4 = go.Figure(data=[
    go.Scatter3d(x=np.real(phi_vals4), y=np.imag(phi_vals4), z=t4, mode='lines', line=dict(color='blue', width=2), name='Спираль времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals4[::100]), y=np.imag(phi_vals4[::100]), z=t4[::100], mode='markers', marker=dict(size=8, color='red', symbol='circle'), name='Резонансы (γ_n)', legendgroup='resonances', showlegend=True),
    go.Scatter3d(x=[np.real(phi_vals4[collapse_idx])], y=[np.imag(phi_vals4[collapse_idx])], z=[t4[collapse_idx]], mode='markers+text', marker=dict(size=8, color='black', symbol='circle'), text=['Коллапс (t ≈ 0)'], textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Коллапс (t ≈ 0)', legendgroup='collapse', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals4[resonance_indices4]), y=np.imag(phi_vals4[resonance_indices4]), z=t4[resonance_indices4], mode='markers+text', marker=dict(size=10, color='blue', symbol='circle'), text=['Эффект H'] * len(resonance_indices4), textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Эффект H', legendgroup='h_effect', showlegend=True),
    go.Scatter3d(x=[0], y=[0], z=[axis_range4 + 2], mode='text', text=['Ось смысла'], textfont=dict(size=14, color='black', family='Arial', weight='bold'), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[axis_range4 + 2, axis_range4 + 0.5], mode='lines', line=dict(color='black', width=2), showlegend=False),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-time_range4, time_range4], mode='lines', line=dict(color='purple', width=5), name='Ось времени', legendgroup='time_axis', showlegend=True),
    go.Scatter3d(x=[0], y=[0], z=[time_range4 + 2], mode='text', text=['Ось времени'], textfont=dict(size=14, color='purple', family='Arial', weight='bold'), showlegend=False)
])

fig4.update_layout(
    scene=dict(
        xaxis_title='Re(φ(t))',
        yaxis_title='Im(φ(t))',
        zaxis_title='Время (с)',
        aspectmode='cube',
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-0.5, 1.5]),
        zaxis=dict(range=[-time_range4 - 3, time_range4 + 3])
    ),
    title=dict(text='Спираль осознания', x=0.5, font=dict(size=16, weight='bold', family='Arial')),
    showlegend=True,
    width=500,
    height=600,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        x=0.0,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    ),
    scene_dragmode='orbit',
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25))
)

print("\nОписание графика 4:")
print("Синяя спираль, сглаженная H, движется вокруг невидимой оси смысла (±60 с). Красные точки — резонансы (γ_n), чёрный маркер — коллапс (t≈0), синие маркеры с 'Эффект H' — влияние H на γ_1, γ_2. Фиолетовая ось времени задаёт направление.")

print("Отображение графика 4: Спираль осознания")
fig4.show()

# График 5: Поток времени

print("\nШаг 8: График 5 (Поток времени)")

t5 = np.linspace(-50, 50, 1000)
phi_vals5 = np.exp(1j * t5) * 0.5 * regularize_factor(t5)
time_range5 = 50
colors5 = np.linspace(0, 1, len(t5))
milestone_times = [-40, 0, 40]
milestone_indices = [np.argmin(np.abs(t5 - t)) for t in milestone_times]
resonance_indices5 = [np.argmin(np.abs(t5 - gamma)) for gamma in resonance_times]

fig5 = go.Figure(data=[
    go.Scatter3d(x=np.real(phi_vals5), y=np.imag(phi_vals5), z=t5, mode='lines', line=dict(color=colors5, colorscale='Plasma', width=3), name='Поток времени', legendgroup='spiral', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals5[milestone_indices]), y=np.imag(phi_vals5[milestone_indices]), z=t5[milestone_indices], mode='markers+text', marker=dict(size=8, color='green', symbol='circle'), text=['Веха времени'] * len(milestone_indices), textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Вехи времени', legendgroup='milestones', showlegend=True),
    go.Scatter3d(x=np.real(phi_vals5[resonance_indices5]), y=np.imag(phi_vals5[resonance_indices5]), z=t5[resonance_indices5], mode='markers+text', marker=dict(size=10, color='blue', symbol='circle'), text=['Эффект H'] * len(resonance_indices5), textposition='middle right', textfont=dict(size=12, color='black', family='Arial', weight='bold'), name='Эффект H', legendgroup='h_effect', showlegend=True),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-time_range5, time_range5], mode='lines', line=dict(color='purple', width=5), name='Ось времени', legendgroup='time_axis', showlegend=True),
    go.Scatter3d(x=[0], y=[0], z=[time_range5 + 2], mode='text', text=['Ось времени'], textfont=dict(size=14, color='purple', family='Arial', weight='bold'), showlegend=False)
])

fig5.update_layout(
    scene=dict(
        xaxis_title='Re(φ(t))',
        yaxis_title='Im(φ(t))',
        zaxis_title='Время (с)',
        aspectmode='cube',
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-time_range5 - 3, time_range5 + 3])
    ),
    title=dict(text='Поток времени', x=0.5, font=dict(size=16, weight='bold', family='Arial')),
    showlegend=True,
    coloraxis_showscale=False,
    width=500,
    height=600,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        x=0.0,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    ),
    scene_dragmode='orbit',
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25))
)

print("\nОписание графика 5:")
print("Спираль с градиентом (Plasma), сглаженная H, символизирует поток времени (±50 с). Зелёные вехи — ключевые моменты, синие маркеры с 'Эффект H' — влияние H на γ_1, γ_2. Фиолетовая ось времени задаёт направление.")

print("Отображение графика 5: Поток времени")
fig5.show()

# График 6: Космос Любона

# Параметры
T = 50
t = np.linspace(-T, T, 1000)
gamma_n = [14.1347, 21.0220, 25.0108]  # γ_1, γ_2 для звёзд, γ_3 для пульсара
omega = 1e3
scale_factor = 2.0
time_range = 50
num_stars = 100  # Для звёздного поля

# Поле Любона
print("\nШаг 9: Вычисление поля phi(t) для графика 6")
def phi_t(t_val, time_shift=0):
    phi = np.exp(1j * (t_val + time_shift)) * np.sin(t_val / 5) * regularize_factor(t_val)
    return phi * scale_factor

# Анимационные кадры
frames_data = []
num_frames = 10
time_shifts = np.linspace(0, 5, num_frames)
for time_shift in time_shifts:
    phi_vals = np.array([phi_t(ti, time_shift) for ti in t])
    frames_data.append(phi_vals)

# Резонансы, пульсар, планеты, спутники, кометы, чёрные дыры
phi_vals = frames_data[0]  # Начальный кадр
abs_phi = np.abs(phi_vals)
resonance_indices = [np.argmin(np.abs(t - gamma)) for gamma in gamma_n[:2]]  # γ_1, γ_2
pulsar_index = [np.argmin(np.abs(t - gamma_n[2]))]  # γ_3
planet_indices, _ = find_peaks(abs_phi, height=0.5, distance=50)  # Планеты
satellite_indices = []
for gamma in gamma_n:
    for delta in [-0.5, 0.5]:
        idx = np.argmin(np.abs(t - (gamma + delta)))
        if idx not in resonance_indices + pulsar_index:
            satellite_indices.append(idx)
comet_indices = np.argsort(abs_phi)[-int(0.1 * len(t)):]  # Топ-10% амплитуд
comet_indices = np.random.choice(comet_indices, size=3, replace=False)  # 3 кометы
black_hole_indices = np.where(abs_phi < 0.1)[0][:3]  # 3 чёрные дыры

# Звёздное поле
np.random.seed(42)
star_x = np.random.uniform(-2, 2, num_stars)
star_y = np.random.uniform(-2, 2, num_stars)
star_z = np.random.uniform(-T, T, num_stars)

# Визуализация
print("Создание графика 6: Космос Любона")
colors = np.linspace(0, 1, len(t))
x_range = [np.min(np.real(phi_vals)) * 1.5, np.max(np.real(phi_vals)) * 1.5]
y_range = [np.min(np.imag(phi_vals)) * 1.5, np.max(np.imag(phi_vals)) * 1.5]

fig6 = go.Figure()

# Звёздное поле
fig6.add_trace(go.Scatter3d(
    x=star_x,
    y=star_y,
    z=star_z,
    mode='markers',
    marker=dict(size=2, color='lightgray', symbol='circle'),
    name='Звёздное поле',
    showlegend=False
))

# Спираль Любона
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals),
    y=np.imag(phi_vals),
    z=t,
    mode='lines',
    line=dict(color=colors, colorscale='Plasma', width=4),
    name='Спираль Любона',
    showlegend=False
))

# Звёзды (γ_1, γ_2)
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[resonance_indices]),
    y=np.imag(phi_vals[resonance_indices]),
    z=t[resonance_indices],
    mode='markers+text',
    marker=dict(size=8, color='red', symbol='cross'),
    text=['Звезда'] * len(resonance_indices),
    textposition='top center',
    textfont=dict(size=12, color='black', family='Arial', weight='bold'),
    name='Звезда',
    legendgroup='resonances'
))

# Пульсар (γ_3)
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[pulsar_index]),
    y=np.imag(phi_vals[pulsar_index]),
    z=t[pulsar_index],
    mode='markers+text',
    marker=dict(size=10, color='blue', symbol='diamond'),
    text=['Пульсар'],
    textposition='bottom center',
    textfont=dict(size=12, color='black', family='Arial', weight='bold'),
    name='Пульсар',
    legendgroup='h_effect'
))

# Планеты
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[planet_indices]),
    y=np.imag(phi_vals[planet_indices]),
    z=t[planet_indices],
    mode='markers+text',
    marker=dict(size=6, color='green', symbol='circle'),
    text=['Планета'] * len(planet_indices),
    textposition='middle right',
    textfont=dict(size=10, color='black', family='Arial', weight='bold'),
    name='Планета',
    legendgroup='planets'
))

# Спутники
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[satellite_indices]),
    y=np.imag(phi_vals[satellite_indices]),
    z=t[satellite_indices],
    mode='markers+text',
    marker=dict(size=4, color='yellow', symbol='circle'),
    text=['Спутник'] * len(satellite_indices),
    textposition='middle left',
    textfont=dict(size=8, color='black', family='Arial', weight='bold'),
    name='Спутник',
    legendgroup='satellites'
))

# Кометы
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[comet_indices]),
    y=np.imag(phi_vals[comet_indices]),
    z=t[comet_indices],
    mode='markers+text',
    marker=dict(size=7, color='cyan', symbol='diamond'),
    text=['Комета'] * len(comet_indices),
    textposition='top center',
    textfont=dict(size=10, color='black', family='Arial', weight='bold'),
    name='Комета',
    legendgroup='comets'
))

# Чёрные дыры
fig6.add_trace(go.Scatter3d(
    x=np.real(phi_vals[black_hole_indices]),
    y=np.imag(phi_vals[black_hole_indices]),
    z=t[black_hole_indices],
    mode='markers+text',
    marker=dict(size=14, color='black', symbol='circle', opacity=0.7),
    text=['Чёрная дыра'] * len(black_hole_indices),
    textposition='bottom center',
    textfont=dict(size=10, color='black', family='Arial', weight='bold'),
    name='Чёрная дыра',
    legendgroup='black_holes'
))

# Ось времени
fig6.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[-time_range, time_range],
    mode='lines',
    line=dict(color='purple', width=5),
    name='Ось времени',
    showlegend=False
))
fig6.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[time_range + 2],
    mode='text',
    text=['Ось времени'],
    textfont=dict(size=14, color='purple', family='Arial', weight='bold'),
    showlegend=False
))

# Анимационные кадры
frames = []
for f, time_shift in enumerate(time_shifts):
    phi_vals = frames_data[f]
    abs_phi = np.abs(phi_vals)
    comet_indices = np.argsort(abs_phi)[-int(0.1 * len(t)):]
    comet_indices = np.random.choice(comet_indices, size=3, replace=False)
    black_hole_indices = np.where(abs_phi < 0.1)[0][:3]
    marker_size = 10 + 2 * np.sin(f * np.pi / num_frames)  # Пульсация пульсара
    frame_data = [
        go.Scatter3d(
            x=star_x,
            y=star_y,
            z=star_z,
            mode='markers',
            marker=dict(size=2, color='lightgray', symbol='circle'),
            name='Звёздное поле'
        ),
        go.Scatter3d(
            x=np.real(phi_vals),
            y=np.imag(phi_vals),
            z=t,
            mode='lines',
            line=dict(color=colors, colorscale='Plasma', width=4),
            name='Спираль Любона'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[resonance_indices]),
            y=np.imag(phi_vals[resonance_indices]),
            z=t[resonance_indices],
            mode='markers+text',
            marker=dict(size=8, color='red', symbol='cross'),
            text=['Звезда'] * len(resonance_indices),
            textposition='top center',
            textfont=dict(size=12, color='black', family='Arial', weight='bold'),
            name='Звезда'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[pulsar_index]),
            y=np.imag(phi_vals[pulsar_index]),
            z=t[pulsar_index],
            mode='markers+text',
            marker=dict(size=marker_size, color='blue', symbol='diamond'),
            text=['Пульсар'],
            textposition='bottom center',
            textfont=dict(size=12, color='black', family='Arial', weight='bold'),
            name='Пульсар'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[planet_indices]),
            y=np.imag(phi_vals[planet_indices]),
            z=t[planet_indices],
            mode='markers+text',
            marker=dict(size=6, color='green', symbol='circle'),
            text=['Планета'] * len(planet_indices),
            textposition='middle right',
            textfont=dict(size=10, color='black', family='Arial', weight='bold'),
            name='Планета'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[satellite_indices]),
            y=np.imag(phi_vals[satellite_indices]),
            z=t[satellite_indices],
            mode='markers+text',
            marker=dict(size=4, color='yellow', symbol='circle'),
            text=['Спутник'] * len(satellite_indices),
            textposition='middle left',
            textfont=dict(size=8, color='black', family='Arial', weight='bold'),
            name='Спутник'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[comet_indices]),
            y=np.imag(phi_vals[comet_indices]),
            z=t[comet_indices],
            mode='markers+text',
            marker=dict(size=7, color='cyan', symbol='diamond'),
            text=['Комета'] * len(comet_indices),
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial', weight='bold'),
            name='Комета'
        ),
        go.Scatter3d(
            x=np.real(phi_vals[black_hole_indices]),
            y=np.imag(phi_vals[black_hole_indices]),
            z=t[black_hole_indices],
            mode='markers+text',
            marker=dict(size=14, color='black', symbol='circle', opacity=0.7),
            text=['Чёрная дыра'] * len(black_hole_indices),
            textposition='bottom center',
            textfont=dict(size=10, color='black', family='Arial', weight='bold'),
            name='Чёрная дыра'
        )
    ]
    frame_data.extend(fig6.data[8:])  # Ось времени
    frames.append(go.Frame(data=frame_data, name=f'frame{f}'))

fig6.frames = frames

# Слайдер
sliders = [
    dict(
        steps=[
            dict(
                method='animate',
                args=[[f'frame{f}'], {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                label=f'{f+1}'
            ) for f in range(num_frames)
        ],
        active=0,
        currentvalue={'prefix': 'Кадр: '},
        pad={'t': 50}
    )
]

# Макет
fig6.update_layout(
    scene=dict(
        xaxis_title='Re(φ(t))',
        yaxis_title='Im(φ(t))',
        zaxis_title='Время (с)',
        aspectmode='cube',
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        zaxis=dict(range=[-time_range - 3, time_range + 3])
    ),
    title=dict(
        text='Космос Любона: Полная космология',
        x=0.5,
        font=dict(size=16, weight='bold', family='Arial')
    ),
    showlegend=True,
    coloraxis_showscale=False,
    width=610,
    height=720,
    margin=dict(l=0, r=100, t=60, b=20),
    legend=dict(
        x=1.0,
        y=1.0,
        xanchor='right',
        yanchor='top',
        font=dict(size=10, color='black', family='Arial'),
        itemsizing='constant',
        itemwidth=90
    ),
    sliders=sliders,
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='Играть',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate',
                        'repeat': True
                    }]
                ),
                dict(
                    label='Пауза',
                    method='animate',
                    args=[[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate'
                    }]
                )
            ],
            pad={'r': 10, 't': 10}
        )
    ],
    scene_dragmode='orbit',
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5)),
    template='plotly_white'
)

print("\nОписание графика 6:")
print("Космос Любона — 3D-визуализация времени как галактической спирали (градиент Plasma) на светло-сером звёздном поле. Две красные звезды (γ_1=14.1347, γ_2=21.0220) — резонансы, центры притяжения, организующие структуру времени, подобно звёздам в галактике. Один синий пульсар (γ_3=25.0108) — эффект H, пульсирующий в анимации, символизируя ритмичные изменения в потоке времени. Зелёные планеты — локальные максимумы |φ(t)|, устойчивые 'обитаемые' зоны времени. Жёлтые спутники — гармоники около γ_n, мелкие детали временной структуры. Голубые кометы (галочки) — случайные пики |φ(t)|, перемещающиеся в анимации, представляющие редкие, хаотические события. Чёрные дыры (плоские окружности) — области с |φ(t)| < 0.1, поглощающие время, создающие 'провалы' в космосе. Фиолетовая ось времени (±50 с) задаёт направление. Слайдер управляет вращением спирали, пульсацией пульсара, движением комет и чёрных дыр. Анимация зациклена.")

print("Отображение графика 6: Космос Любона")
fig6.show()

print("* График сохранён как lubon_cosmos_full.html")
fig6.write_html("lubon_cosmos_full.html")
