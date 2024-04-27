# llmft



```python 

probs = F.softmax(outputs.loss['logits'][2].cpu() , dim=-1).detach().numpy()
from matplotlib.colors import Normalize

# Desired colorbar maximum (k)
k = .01

# Visualize probabilities for the first 10 tokens at all sequence positions
plt.figure(figsize=(10, 8))
# Use Normalize to set the color scale from 0 to k
norm = Normalize(vmin=0, vmax=k)
cmap = plt.get_cmap('viridis')  # You can choose any colormap that fits your needs

# Create the heatmap with normalization
cax = plt.imshow(probs, aspect='auto', cmap=cmap, norm=norm)

# Create a colorbar with the correct scaling
cbar = plt.colorbar(cax)  # Ensure ticks cover the range from 0 to k
cbar.set_label('Probability Scale')

plt.title('Probability Distribution Over First 10 Tokens in the Sequence')
plt.xlabel('Token Index')
plt.ylabel('Sequence Position')
plt.show()

```