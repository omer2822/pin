import numpy as np

def split_step_solver(psi0, x, dt, steps, kappa):
    """
    פותר משוואת שרדינגר לא-ליניארית בשיטת Split-Step
    psi0: מצב התחלתי (מערך נומרי)
    x: ציר המרחב
    dt: צעד הזמן
    steps: מספר הצעדים הכולל
    kappa: מקדם הלא-ליניאריות
    """

    N = len(x)
    L = x[-1] - x[0]

    k = 2 * np.pi * np.fft.fftfreq(N, d=(L/N))

    operator_dispersion = np.exp(-1j * (k**2) * (dt / 2))

    psi = np.array(psi0, dtype=complex)

    for _ in steps:
        psi_freq = np.fft.fft(psi)
        psi_freq *= operator_dispersion
        psi = np.fft.ifft(psi_freq)

        operator_nonlinear = np.exp(-1j * kappa * np.abs(psi)**2 * dt)
        psi *= operator_nonlinear

        psi_freq = np.fft.fft(psi)
        psi_freq *= operator_dispersion
        psi = np.fft.ifft(psi_freq)

    return psi


