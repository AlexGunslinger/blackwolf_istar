from sys import intern
from .external.module import module_job_overview as external_job_overview
from .internal.module import module_job_overview as internal_job_overview
                  
def module_job_overview(Proj):
    if Proj.registration_type == 'External scans':
        return external_job_overview(Proj)
    else:
        return internal_job_overview(Proj)