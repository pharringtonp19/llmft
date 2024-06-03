### **Purpose**
These notebooks highlight that language models can exploit information in the text which is (A) informative about who is a complier and (B) something that the research knows is informative of who is a complier.

### **Setp Up**
 1. We generate synthetic observations by drawing features $x = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]$
 2. Using an [anthropic model](https://www.anthropic.com/api), and the following prompt, we map $x$ into text, $t^*(x)$. This gives us a set of numerical control vectors
    which we can pass as inputs to linear and feed-forward models as well as a collection text that we can pass as inputs to the language model. 

    ```python
      Task: Write a paragraph description of a tenant in their {age_group} who is currently {overdue_phrase} ${rent:.0f}. 
      Mention that they have {health}, live in a {living_situation} {voucher}, have been living there for {months} months, and have {pets}. 
      Include some details about their {roomate_status} who {contribute_status} to the rent. Also mention somewhere that {additional_detail}
      
      Description: The tenant is a """
    ```
4. The instrumental variable is randomly assigned so we don't need to control for any features for identification reasons. 
5. A key design choice in this simulation is that (A) the first stage depends heavily on both the type of disability $(x_4)$ and housing voucher status $(x_8)$ and (B) the research knows (from conversations with legal aid lawyers) that tenants with vouchers and disabilities were often prioritized over other tenants. Of course, there are a number of different forms of disability -- Physical, Mental, Developmental, Temporary, and Chronic Health. And the reseacher doesn't know exactly how legal aid lawyers might weigh each type or the extent to which tenants with these characteristics will follow through on the takeup of legal assistancee. So the researcher develops a list of dissabilities and classifies creates one-hot representations of the dissabilities for each text. 
   ```python
   def fstage(var1, var2, var3, var4, var5, var6, var7, var8, var9):
     return .35*(severity_indicator[var5]) + 0.35*var9 + .1
   ```
5. We then evaluate a fine-tuned LLM, a feed-forward neural network, a linear model with no controls, and a linear model which does control for $x_4$ contradicting what we said before but we label it the "oracle model" so it's not really a contradiction.

### **Why Might this Work**
Representational Learning
