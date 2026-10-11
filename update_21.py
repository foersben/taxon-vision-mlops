with open('docs/latex/technical_report/chapters/21_quality_assurance_testing_matrix_and_code_invariants.tex', 'r') as f:
    content = f.read()

content = content.replace("	exttt{", "\\texttt{")

with open('docs/latex/technical_report/chapters/21_quality_assurance_testing_matrix_and_code_invariants.tex', 'w') as f:
    f.write(content)
