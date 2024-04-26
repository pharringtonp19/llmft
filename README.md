# llmft



```python 

ps = np.linspace(0.01, 0.99)
ys0 = -1.0*class_weights[0].cpu()*np.log(ps)
ys1 = -1.0*class_weights[1].cpu()*np.log(ps)
plt.plot(ps, ys0)
plt.plot(ps, ys1)
plt.show()


ps = np.linspace(0.01, 0.99)
for gamma in [0, 2]:
    ys0 = -1*(1-ps)**gamma * np.log(ps)
    plt.plot(ps, ys0, label=gamma)
plt.legend()
plt.show()

```