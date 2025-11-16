import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import get_spectrum3

def show_one_graph(x_in,y_in,startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint = len(x_in)

    fig = go.Figure(data=go.Scatter(x=x_in[startpoint:endpoint], y=y_in[startpoint:endpoint], mode='lines+markers'))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=600, height=400
    )
    fig.show()
    
def plot(x_in,y_in,point=None, delta=None):
    leng=len(x_in)

    if (point==None):
        point = int(leng/2)

    if (delta==None or delta>=leng):
        delta = int(leng/2)

    plt.plot(x_in[point-delta:point+delta], y_in[point-delta:point+delta])


def show_I_U(t,I,U, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint = len(I)

    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    plt.plot(t[startpoint:endpoint] * 1e3, U[startpoint:endpoint] , label='U(t)', color='blue')
    plt.title('U')
    plt.ylabel('(V)')
    plt.grid()

    plt.subplot(4, 1, 1)
    plt.plot(t[startpoint:endpoint] * 1e3, U[startpoint:endpoint] , label='U(t)', color='blue')
    plt.title('U')
    plt.ylabel('(V)')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(t[startpoint:endpoint]  * 1e3, I[startpoint:endpoint]  * 1e3, label='I_total(t)', color='red')
    plt.title('I')
    plt.xlabel('time ms')
    plt.ylabel('(mA)')
    plt.grid()

def show_I_U_2(t,I,U, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint = len(I)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(t[startpoint:endpoint] * 1e3, U[startpoint:endpoint] , color='blue')
    ax1.set_title("U(t)")
    ax1.grid(True)

    F,V=get_spectrum3(t,U)
    ax2.plot(F,V, 'blue')
    ax2.set_title("F(U)")
    ax2.grid(True)

    ax3.plot(t[startpoint:endpoint]  * 1e3, I[startpoint:endpoint]  * 1e3, color='red')
    ax3.set_title("I(t)")
    ax3.grid(True)

    F,V=get_spectrum3(t,I)
    ax4.plot(F,V, 'red')
    ax4.set_title("F(I)")
    ax4.grid(True)

def show_I_U_3(t,I,U, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint = len(I)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("U(t)", "F(U)", "I(t)", "F(I)"),
        column_widths=[0.7, 0.3],   # относительная ширина столбцов
        row_heights=[0.5, 0.5],     # относительная высота строк
        horizontal_spacing=0.05,    # расстояние между колонками (0 = нет)
        vertical_spacing=0.07       # расстояние между строками
    )

    fig.add_trace(go.Scatter(x=t[startpoint:endpoint], y=U[startpoint:endpoint]), row=1, col=1)
    F,V=get_spectrum3(t,U)
    fig.add_trace(go.Scatter(x=F, y=V), row=1, col=2)
    fig.add_trace(go.Scatter(x=t[startpoint:endpoint], y=I[startpoint:endpoint]), row=2, col=1)
    F,V=get_spectrum3(t,I)
    fig.add_trace(go.Scatter(x=F, y=V), row=2, col=2)

    # Минимизируем внешние отступы (по краям всего рисунка)
    fig.update_layout(
        showlegend=False,
        height=600,
        width=900,
        margin=dict(l=30, r=30, t=40, b=30)
    )

    fig.show()