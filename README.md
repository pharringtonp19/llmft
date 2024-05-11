# llmft

### **Purpose**
Producing causal effects by fine-tuning Large Language Models on text data. 

### **Example**
In the [Right to Counsel at Scale](https://github.com/pharringtonp19/papers/blob/main/The_Right_to_Counsel_at_Scale_latest.pdf), we estimate the impact of legal representation (for those facing eviction) on housing court outcomes and housing stability more importantly. We do so via an instrumental variable approach where the instrument is whether free legal representation is available in the tenant's zip code. 

$$y_i = \beta(\big(\mathbb{E}[D_i \vert X_i, Z_i] - \mathbb{E}[D_i \vert X_i]\big)$$

We estimate the conditional expectation functions via fine-tuned LLMs where $X_i$ is the landlord's complaint against the tenant.
