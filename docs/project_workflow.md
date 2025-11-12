# Project workflow

**1. Version Control (Git)**
Each team member develops on their feature branch and merges through pull requests.

**2. Jupyter Workflow**

Each subsystem can be developed and tested in its own notebook under /notebooks/.
Only the final integrated notebook (Final_Report.ipynb) should be cleaned and used for submission.

## Repository structure

```text
.
├── docs/                           # Helpful documents
│ ├── git_cheat_sheet.pdf 
│ ├── markdown_cheat_sheet.md 
│ └── project_workflow.md 
├── notebooks/                      # Jupyter Notebooks for development and report
| ├── ...
│ └── final_report.ipynb            # Final submission notebook
├── src/                            # Core source code (only .py or .aseba files)
│ ├── computer_vision/ 
│ ├── global_navigation/ 
│ ├── local_navigation/ 
│ ├── motion_control/ 
│ ├── pose_estimation/ 
│ ├── utils/ 
│ └── main.py 
├── tests/                          # Optional
├── .gitignore 
└── README.md                       # Main README (project overview)
```