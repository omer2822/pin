Sprint A — Make current science valid

Phases 8–9 + provenance + Phase-7 fairness.

Sprint B — Mathematical foundation

להוסיף:

SpectralOperatorDomain
eigenvalues()
analyze()
synthesize()
apply_function_of_operator()

ולהוציא NLS physics מתוך pde_residual.py ל־EvolutionEquation.


Sprint C — Dirichlet

DST-I + mass/quadrature correctness + existing C1 on both domains.

בלי NN חדש עדיין.

זה חשוב: קודם להוכיח שה־abstraction נכון.

Sprint D — C4

Variational nonlocal phase.

וה־headline experiment:

$$ C2\;vs\;C4. $$
Sprint E — physics-loss factorial

data/CN/strong/rollout/Hamiltonian/spectral-band.

הקובץ שהבאת גם מזהה בצדק שה־CN residual הנוכחי כבר מכניס property מבני, ולכן זה ablation חשוב.

Sprint F — C5

Learn:

$$ f_\theta(A) $$

עם real generator ו־unitary exponential.

Sprint G — variable time
$$ \Phi_t,\quad \Phi_0=I,\quad \Phi_{t+s}\simeq\Phi_t\Phi_s. $$
Sprint H — 2D

רק אז SpectralConvNd.


לא C4 עדיין.

הייתי נותן לו:

Refactor SPNO around basis-independent spectral functional calculus and equation/domain separation without changing any existing periodic experiment outputs. Add a Dirichlet spectral domain as the first new implementation, but do not add new model behavior yet.

כי אם הבסיס הזה נקי, C4/C5 יהיו פי כמה יותר פשוטים.


