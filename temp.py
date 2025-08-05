import numpy as np
import matplotlib.pyplot as plt


temperature = np.array((23,45,34,22,12,23,14,15,20,28,33,12,44,34,35,36,38,40,44,43,38,39,37,35,32,30,28,27,25,22))

mean_temp = np.mean(temperature)
max_temp = np.max(temperature)
min_temp = np.min(temperature)
std_temp = np.std(temperature)

print(f"The average temperature is:{mean_temp}")
print(f"The maximum temperature is:{max_temp}")
print(f"The minimum temperature is:{min_temp}")
print(f"The standard temperature is:{std_temp}")

plt.plot(temperature , marker='o')
plt.title("Line chart of temperature")
plt.xlabel("Day")
plt.ylabel("Temperatures (celcius)")
plt.show()