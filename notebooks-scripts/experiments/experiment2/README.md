### **Purpose**
These notebooks hightlight that language models can exploit information in the text which is (A) informative about who is a complier and (B) something that the researcher would not have decided to represent as a numerical control variable.

### **Setp Up**
 1. We generate synthetic observations by drawing features $x = [x_0, x_1, x_2, x_3, x_4]$
 2. Using an [anthropic model](https://www.anthropic.com/api), and the following prompt, we map $x$ into text, $t^*(x)$. This allows us to have numerical features
    which we can pass as inputs to linear and feed-forward models as well as text that we can pass to to the language model. 
  ```python
  def get_promptv3(i, x):
  """Generates a prompt for writing a paragraph about a tenant.
  
  Args:
    i: Random seed for reproducibility.
    x: List of features for the tenant.
  
  Returns:
    A formatted string containing the prompt.
  """
  age_group = "mid-20s" if x[0] == 1 else "mid-30s"
  living_situation = "small appartment complex" if x[1] == 1 else "large appartment complex"
  pets = "a dog" if x[2] == 1 else "no pets"
  
  return f"""random seed: {i}
  Task: Write a paragraph description of a tenant in their {age_group} who is currently behind on rent for ${x[3]:.0f}. 
  Mention that they have relatively {'good health' if x[4] == 1 else 'poor health'}, live in a {living_situation}, and have {pets}.  
  
  Description: The tenant is a """
  ```
3. The instrumental variable is randomly assigned so we don't need to control for any features for identification reasons. 
4. A key design choice in this simulation is that (A) the first stage depends heavily on $x_4$ and (B) $x_4$ is not passed as input to the linear and feed-forward models.
Or put another way, there is a "some information" in the text which is (A) highly indicative of who is a complier and (B) is not a feature that a researcher would have chosen apriori to select as a control variable. In the above simulation $x_4$ corresponds to health status.
  ```python
  def fstage(var0, var1, var2, var3, var4):
      return (1.0-var4)
  ```
5. We then evaluate a fine-tuned LLM, a feed-forward neural network, a linear model with no controls, and a linear model which does control for $x_4$ contradicting what we said before but we label it the "oracle model" so it's not really a contradiction. 
