import matplotlib.pyplot as plt

def show_I_U(t,I,U, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint = len(I)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t[startpoint:endpoint] * 1e3, U[startpoint:endpoint] , label='U(t)', color='blue')
    plt.title('U')
    plt.ylabel('(V)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t[startpoint:endpoint]  * 1e3, I[startpoint:endpoint]  * 1e3, label='I_total(t)', color='red')
    plt.title('I')
    plt.xlabel('time ms')
    plt.ylabel('(mA)')
    plt.grid()