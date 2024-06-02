### **Purpose**
These notebooks hightlight that language models can exploit information in the text which is (A) informative about who is a complier and (B) something that the researcher would not have decided to represent as a numerical control variable.

### **Setp Up**
 1. We generate synthetic observations by drawing features $x = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]$
 2. Using an [anthropic model](https://www.anthropic.com/api), and the following prompt, we map $x$ into text, $t^*(x)$. This gives us a set of numerical control vectors
    which we can pass as inputs to linear and feed-forward models as well as a collection text that we can pass as inputs to the language model. 
  ```
    Task: Write a paragraph description of a tenant in their {age_group} who is currently {overdue_phrase} ${x[3]:.0f}. 
    Mention that they are in relatively {health}, live in a {living_situation}, have been living there for {months} months, and have {pets}. 
    Include some details about their {roomate_status} who {contribute_status} to the rent. Also mention somewhere that {additional_detail}
    
    Description: The tenant is a """
  ```
3. The instrumental variable is randomly assigned so we don't need to control for any features for identification reasons. 
4. A key design choice in this simulation is that (A) the first stage depends heavily on $x_4$ and (B) $x_4$ is not passed as input to the linear and feed-forward models.
Or put another way, there is a "some information" in the text which is (A) highly indicative of who is a complier and (B) is not a feature that a researcher would have chosen apriori to select as a control variable. In the above simulation $x_4$ corresponds to health status.
  ```python
  def fstage(var1, var2, var3, var4, var5, var6, var7, var8):
    return (1.0-var5)
  ```
5. We then evaluate a fine-tuned LLM, a feed-forward neural network, a linear model with no controls, and a linear model which does control for $x_4$ contradicting what we said before but we label it the "oracle model" so it's not really a contradiction. 
