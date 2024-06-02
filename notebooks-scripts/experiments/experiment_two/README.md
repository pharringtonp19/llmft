### **Purpose**
These notebooks hightlight that language models can exploit information in the text which is (A) informative about who is a complier and (B) something that the research knows is informative of who is a complier.

### **Setp Up**
 1. We generate synthetic observations by drawing features $x = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]$
 2. Using an [anthropic model](https://www.anthropic.com/api), and the following prompt, we map $x$ into text, $t^*(x)$. This gives us a set of numerical control vectors
    which we can pass as inputs to linear and feed-forward models as well as a collection text that we can pass as inputs to the language model. 

    ```python
    def get_promptv3(i, x):
      # Add variability
      age_group = age_groups[x[0]]
      living_situation = living_situations[x[1]]
      pets = pets_options[x[2]]
      rent = x[3]
      health = disabilities[x[4]]
      months = x[5]
      roomate_status = roomate_statuses[x[6]]
      contribute_status = contribute_statuses[x[7]]
      voucher = voucher_status[x[8]]
  
      # Add some noise with random synonyms or additional details
      overdue_phrase = random.choice(overdue_phrases)
      additional_detail = random.choice(additional_details)
  
      return f"""random seed: {i}
      Task: Write a paragraph description of a tenant in their {age_group} who is currently {overdue_phrase} ${rent:.0f}. 
      Mention that they have {health}, live in a {living_situation} {voucher}, have been living there for {months} months, and have {pets}. 
      Include some details about their {roomate_status} who {contribute_status} to the rent. Also mention somewhere that {additional_detail}
      
      Description: The tenant is a """
    ```
4. The instrumental variable is randomly assigned so we don't need to control for any features for identification reasons. 
5. A key design choice in this simulation is that (A) the first stage depends heavily on both $the type of disability (x_4) and housing voucher status (x_8)$ and (B) the research knows (from conversations with legal aid lawyers) that tenants with vouchers and disabilities were often prioritized over other tenants.
Or put another way, there is a "some information" in the text which is (A) highly indicative of who is a complier and (B) is not a feature that a researcher would have chosen apriori to select as a control variable. In the above simulation $x_4$ corresponds to health status.
  ```python
  def fstage(var1, var2, var3, var4, var5, var6, var7, var8):
    return (1.0-var5)
  ```
5. We then evaluate a fine-tuned LLM, a feed-forward neural network, a linear model with no controls, and a linear model which does control for $x_4$ contradicting what we said before but we label it the "oracle model" so it's not really a contradiction. 
