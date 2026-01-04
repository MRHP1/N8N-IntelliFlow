
import os

file_path = r'c:\Users\risya\Downloads\Kuliah\Semester5\TKTI\Papers\paper_compressed.tex'

new_bib = r"""\begin{thebibliography}{00}

% --- 1. Table I Citations ---

\bibitem{shamim2025_costreview}
M. M. I. Shamim \textit{et al.},
``Advancement of Artificial Intelligence in Cost Estimation for Project Management Success: A Systematic Review,''
\textit{Modelling}, vol. 6, no. 2, Art. 35, 2025,
doi: 10.3390/modelling6020035.

\bibitem{sousa2023_task_effort}
T. Sousa \textit{et al.}, ``Applying Machine Learning to Estimate the Effort and Duration of Individual Tasks in Software Projects,''
\textit{IEEE Access}, vol. 11, pp. 89933--89946, 2023,
doi: 10.1109/ACCESS.2023.3287537.

\bibitem{prinz2021_lowcode_slr}
N. Prinz, C. Rentrop, and M. Huber, ``Low-Code Development Platforms -- A Literature Review,''
in \textit{Proc. Americas Conf. Inf. Syst. (AMCIS)}, 2021.

\bibitem{yasmin2024_airflow_wac}
J. Yasmin, J. Wang, Y. Tian, and B. Adams, ``An Empirical Study of Developers' Challenges in Implementing Workflows as Code: A Case Study on Apache Airflow,''
\textit{arXiv preprint} arXiv:2406.00180, 2024.

\bibitem{perezcastillo2024_sprint_velocity}
Y. J. P\'erez Castillo, S. D. Orantes Jim\'enez, and P. O. Letelier Torres,
``Sprint Management in Agile Approach: Progress and Velocity Evaluation Applying Machine Learning,''
\textit{Information}, vol. 15, no. 11, Art. 726, 2024,
doi: 10.3390/info15110726.

\bibitem{diamantopoulos2023_issue_assignment}
T. Diamantopoulos, N. Saoulidis, and A. Symeonidis,
``Automated Issue Assignment Using Topic Modelling on Jira Issue Tracking Data,''
\textit{IET Software}, vol. 17, no. 3, pp. 333--344, 2023,
doi: 10.1049/sfw2.12129.

% --- 2. Introduction Citations ---

\bibitem{verwijs2023_scrum}
C. Verwijs and D. Russo, ``A Theory of Scrum Team Effectiveness,''
\textit{ACM Trans. Softw. Eng. Methodol.}, vol. 32, no. 3, Art. 74, Apr. 2023,
doi: 10.1145/3571849.

\bibitem{gemino2021_project_success}
A. Gemino, B. H. Reich, and R. Serrador, ``Agile, traditional, and hybrid approaches to project success: Is hybrid a poor second choice?''
\textit{Project Management Journal}, vol. 52, no. 2, pp. 161--175, 2021,
doi: 10.1177/8756972820973082.

\bibitem{gibbs2023_wfh_productivity}
M. Gibbs, F. Mengel, and C. Siemroth, ``Work from Home and Productivity: Evidence from Personnel and Analytics Data on IT Professionals,''
\textit{J. Political Economy Microeconomics}, vol. 1, no. 2, pp. 235--275, 2023,
doi: 10.1086/721803.

\bibitem{venkatesh2020_work_exhaustion}
V. Venkatesh, J. Y. L. Thong, and F. K. Y. Chan, ``How Agile Software Development Methods Reduce Work Exhaustion: Insights on Role Perceptions and Organizational Skills,''
\textit{Information Systems Journal}, vol. 30, no. 5, pp. 829--859, 2020,
doi: 10.1111/isj.12282.

\bibitem{zhao2021_trigger_action}
D. Zhao \textit{et al.}, ``What Can We Learn from a Dataset of If-This-Then-That Recipes?''
in \textit{Proc. CHI Conf. Human Factors in Computing Systems (CHI)}, 2021,
doi: 10.1145/3411764.3445567.

\bibitem{mahdi2021_spm_slr}
N. Mahdi \textit{et al.}, ``Machine Learning for Software Project Management: A Systematic Literature Review,''
\textit{Applied Sciences}, vol. 11, no. 23, 2021.

\bibitem{cabral2023_ensemble_effort}
J. T. H. d. A. Cabral, R. Gomes, and A. Mendes-Moreira,
``Ensemble Effort Estimation: An Updated and Extended Systematic Literature Review,''
\textit{J. Syst. Softw.}, vol. 195, Art. 111542, 2023,
doi: 10.1016/j.jss.2022.111542.

% --- 3. Related Work Citations ---

\bibitem{gupta2023_action_suggestions}
S. Gupta \textit{et al.}, ``Personalized Action Suggestions in Low-Code Automation Platforms,''
in \textit{Proc. IEEE/ACM Int. Conf. Softw. Eng. Companion (ICSE-Companion)}, 2023,
doi: 10.1109/ICSE-Companion58688.2023.00100.

\bibitem{choetkiertikul2025_sprint2vec}
M. Choetkiertikul \textit{et al.},
``Sprint2Vec: A Deep Characterization of Sprints in Iterative Software Development,''
\textit{IEEE Trans. Softw. Eng.}, vol. 51, no. 1, pp. 220--242, 2025,
doi: 10.1109/TSE.2024.3509016.

\bibitem{ramessur2021_sprint_effort}
S. Ramessur and S. D. Nagowah,
``Software Effort Estimation in a Sprint Using Machine Learning Regression Techniques,''
\textit{Int. J. Inf. Technol.}, vol. 13, pp. 1--10, 2021,
doi: 10.1007/s41870-021-00669-z.

\bibitem{yalciner2024_storypoint_sbert}
B. Yal\c{c}{\i}ner and D. Baturay,
``Enhancing Agile Story Point Estimation: Integrating Deep Learning, Machine Learning, and Natural Language Processing with SBERT and Gradient Boosted Trees,''
\textit{Applied Sciences}, vol. 14, no. 16, Art. 7305, 2024,
doi: 10.3390/app14167305.

\bibitem{guo2020_bug_triage}
S. Guo \textit{et al.}, ``Developer Activity Motivated Bug Triaging: Via Convolutional Neural Network,''
\textit{Neural Process. Lett.}, vol. 52, pp. 1--18, 2020,
doi: 10.1007/s11063-020-10213-y.

\bibitem{zhang2023_susrec}
W. Zhang, J. Zhao, R. Peng, S. Wang, and Y. Yang,
``SusRec: An Approach to Sustainable Developer Recommendation for Bug Resolution Using Multimodal Ensemble Learning,''
\textit{IEEE Trans. Reliability}, vol. 72, no. 1, pp. 61--78, 2023,
doi: 10.1109/TR.2022.3176733.

% --- 4. Methods / Data / LBE ---

\bibitem{koralage2023_velocity}
R. U. Koralage, ``Novel Approach to Estimate Velocity for Agile Scrum Sprint Planning,''
Preprint, Dec. 2023.

\bibitem{mahmud2022_risk_slr}
M. S. Mahmud \textit{et al.}, ``Software Project Risk Prediction Using Machine Learning: A Systematic Literature Review,''
\textit{Applied Sciences}, vol. 12, no. 4, 2022.

\bibitem{nastos2025_resolution_time}
D.-N. Nastos, T. Diamantopoulos, D. Tosi, M. Tropeano, and A. Symeonidis,
``Towards an Interpretable Analysis for Estimating the Resolution Time of Software Issues,''
\textit{arXiv preprint} arXiv:2505.01108, 2025.

\bibitem{brar2022_effort_slr}
A. Brar and A. Nandal,
``A Systematic Literature Review on Software Effort Estimation Using Machine Learning Techniques,''
\textit{in Proc. (IEEE-indexed) Conf.}, 2022.

\bibitem{lbe_9234227}
R. S. Dewi and R. Sarno, ``Software Effort Estimation Using Early COSMIC to Substitute Use Case Weight,''
in \textit{Proc. 2020 International Seminar on Application for Technology of Information and Communication (iSemantic)},
pp. 214--219, 2020, doi: 10.1109/iSemantic50169.2020.9234227.

\bibitem{lbe_11229547}
K. C. Febryanto, S. C. Hidayati, and R. Sarno, ``Multi-Dimensional Quality Assessment of Synthetic Data across ERP Modules,''
in \textit{Proc. 2025 IEEE International Conference on Artificial Intelligence and Mechatronics Systems (AIMS)},
pp. 1--6, 2025, doi: 10.1109/AIMS66189.2025.11229547.

\bibitem{lbe_11019730}
Z. L. Putra, R. Sarno, A. F. Septiyanto, R. Januar Akbar, and F. Taufany,
``Evaluating The Modularity of Domain-Driven Design Approach: A Case Study of Academic Information System,''
in \textit{Proc. 2025 International Conference on Computer Sciences, Engineering, and Technology Innovation (ICoCSETI)},
pp. 507--512, 2025, doi: 10.1109/ICoCSETI63724.2025.11019730.

\end{thebibliography}"""

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = r"\begin{thebibliography}{00}"
end_marker = r"\end{thebibliography}"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_marker)
    new_content = content[:start_idx] + new_bib + content[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Bibliography updated successfully.")
else:
    print("Could not find start/end markers.")
