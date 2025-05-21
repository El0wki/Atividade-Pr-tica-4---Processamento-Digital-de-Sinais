import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 5]
plt.rcParams.update({'font.size': 18})

# Cria um sinal sintético com duas frequências: 120Hz e 50Hz
dt = 0.0005                                      # Espaçamento temporal utilizado
t = np.arange(0,1,dt)                            # Vetor tempo
f = (
    1.2 * np.sin(2*np.pi*40*t) +
    0.7 * np.sin(2*np.pi*90*t) +
    0.5 * np.sin(2*np.pi*130*t)   # ruído gaussiano
) # Soma das senoides com as frequências citadas
f_limpo = f                                      # Armazenamos o sinal limpo
f = f + 2.5*np.random.randn(len(t))              # Adiona ruído ao sinal anterior

# # Ploting
plt.plot(t, f, color='r', linewidth=1.0, label='Ruidoso')
plt.plot(t, f_limpo, color='k', linewidth=1.5, label='Limpo')
plt.xlim(t[0],t[-1])
plt.legend()
plt.show()

## Computando a Fast Fourier Transform (FFT)

n = len(t)
fhat = np.fft.fft(f,n)                     # Computa a FFT
DEP = fhat * np.conj(fhat) / n             # Espectro de potência (potÇencia por freq)
freq = (1/(dt*n)) * np.arange(n)           # Cria um eixo de frequências, a partir do eixo temporal
L = np.arange(1,np.floor(n/5),dtype='int') # Plota apenas as primeiras frequências

# Ploting
plt.plot(freq[L], DEP[L], color='c', linewidth=1., label='FFT')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
plt.show()


# Defina as bandas de interesse (com tolerância de ±5 Hz)
bandas = [(35, 45), (85, 95), (125, 135)]
mascara = np.zeros_like(freq, dtype=bool)
for fmin, fmax in bandas:
    mascara |= ((freq >= fmin) & (freq <= fmax))

# Aplicando a máscara na FFT
fhat_filtrado = np.zeros_like(fhat)
fhat_filtrado[mascara] = fhat[mascara]

# Sinal filtrado no tempo
ffilt = np.fft.ifft(fhat_filtrado)


# Plotando o sinal filtrado e o limpo no tempo
plt.plot(t, ffilt.real, color='g', linewidth=1.5, label='Filtrado')
plt.plot(t, f_limpo, color='k', linewidth=1.5, label='Limpo')
plt.xlim(t[0], t[-1])
plt.legend()
plt.title('Sinal no domínio do tempo')
plt.show()

#Plotando o espectro do sinal filtrado
fhat_filt = np.fft.fft(ffilt, n)
DEP_filt = fhat_filt * np.conj(fhat_filt) / n

plt.plot(freq[L], DEP[L], color='c', linewidth=1., label='FFT Ruidoso')
plt.plot(freq[L], DEP_filt[L], color='g', linewidth=1., label='FFT Filtrado')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
plt.title('Espectro de frequência')
plt.show()

# Código do filtro (comentado para referência futura)
'''
## Usando a potência para filtrar o ruído
indices = DEP > 100       # Encontra as potências maiores que o threshold
DEPlimpo = DEP * indices  # Zera todos os outros indices

# Ploting
plt.plot(freq[L], DEPlimpo[L], color='k', linewidth=1.5, label='FFT filtrado')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
plt.show()

fhat_filtrado = indices * fhat     # Zera todos os coeficientes da transformada de fourier
ffilt = np.fft.ifft(fhat_filtrado) # FFT inversa para o sinal filtrado
# Ploting
plt.plot(t, ffilt, color='r', linewidth=1.5, label='Sinal filtrado')
plt.plot(t, f_limpo, color='k', linewidth=1.5, label='Sinal Original (Limpo)')
plt.xlim(t[0], t[-1])
plt.ylim(-4, 4)
plt.legend()
plt.show()
'''