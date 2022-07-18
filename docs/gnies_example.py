import sempler
import sempler.generators
import gnies

# Generate a random SCM using sempler
W = sempler.generators.dag_avg_deg(10, 2.1, 0.5, 1)
scm = sempler.LGANM(W, (0, 0), (1, 2))

# Generate interventional data
data = [
    scm.sample(n=1000),
    scm.sample(n=1000, noise_interventions={1: (0, 11)}),
    scm.sample(n=1000, noise_interventions={2: (0, 12), 3: (0, 13)}),
]

# Run GnIES
gnies.fit(data, approach="rank", direction="backward", debug=1)

# Run GnIES
gnies.fit(data, debug=1)
