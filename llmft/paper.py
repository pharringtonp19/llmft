def generate_recall_table(data):
    latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{c|c|c}\n"
    latex_table += "\\hline\n"
    latex_table += "Epoch & Recall Class 1 & Recall Class 2 \\\\\n\\hline\n"
    
    for i, (recall1, recall2) in enumerate(data):
        latex_table += f"{i + 1} & {recall1} & {recall2} \\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Recall of Two Classes Over Epochs}\n"
    latex_table += "\\end{table}\n"
    return latex_table