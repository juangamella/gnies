import sempler, sempler.generators
import gnies

# Generate a random SCM using sempler
W = sempler.generators.dag_avg_deg(10, 2.1, 0.5, 1, random_state=42)
scm = sempler.LGANM(W, (0, 0), (1, 2), random_state=42)

# Generate interventional data
n = 1000
data = [
    scm.sample(n, random_state=42),
    scm.sample(n, noise_interventions={1: (0, 11)}, random_state=42),
    scm.sample(n, noise_interventions={2: (0, 12), 3: (0, 13)}, random_state=42),
]

# Run GnIES
_score, icpdag, I = gnies.fit(data, approach='rank')
print(icpdag, I)

# Output:
# [[0 1 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 1 0 0 1]
#  [1 0 1 0 0 0 0 0 0 0]
#  [0 1 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 1]
#  [0 1 0 1 0 0 1 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]] {1, 2, 3}
