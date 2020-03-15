import tensorflow as tf
import tensorflow_probability as tfp
p = [10,20]
dist = tfp.distributions.Multinomial(total_count=1000, probs=p)
print(dist.sample(10))

