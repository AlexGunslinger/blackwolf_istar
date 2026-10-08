from .external.module import module_registration as external_module_registration
from .internal.with_floating_roof.module import module_registration as internal_fr_module_registration
from .internal.without_floating_roof.module import module_registration as internal_no_fr_module_registration

def module_registration(Proj):
    if Proj.registration_type == 'Internal scans':
        if Proj.floating_roof:
            return internal_fr_module_registration(Proj)
        else:
            return internal_no_fr_module_registration(Proj)
    else:
        return external_module_registration(Proj)