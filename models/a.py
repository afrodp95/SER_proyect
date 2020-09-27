y = ravdess_data[100,:].reshape(-1)
x = list(range(0,len(y)))

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x[1:],y[1:])
ax.set_xlabel('Time')
ax.set_ylabel('Mean MFCC')
plt.show()
