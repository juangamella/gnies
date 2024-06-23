import sempler
import sempler.generators
import gnies

# Generate a random SCM using sempler
W = sempler.generators.dag_avg_deg(10, 2.1, 0.5, 1)
scm = sempler.LGANM(W, (0, 0), (1, 2))

# Generate interventional data
data = [
    scm.sample(n=500),
    scm.sample(n=500, noise_interventions={1: (5, 2)}),
    scm.sample(n=500, noise_interventions={2: (3, 2), 3: (4, 1.7)}),
]

# Run GnIES
_score, icpdag, I = gnies.fit(data, center=False, debug=1)
print(icpdag, I)
# Output:
# [[0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 0 0 0 0 1 1]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 1 0 0]
#  [0 0 1 0 0 0 0 0 0 1]
#  [0 1 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]] {1, 2, 3}
