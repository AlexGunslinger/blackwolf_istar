import streamlit as st
from .module_cylinder_extraction import module_cylinder_extraction
from .module_rough_alignment import module_rough_alignment
from .module_fine_alignment import module_fine_alignment
def module_registration(Proj):
    st.markdown('<font color="blue">MAIN PROCESSES</font>', unsafe_allow_html=True)
    
    if Proj.pre_aligned is not True:
        module_cylinder_extraction(Proj)        
        module_rough_alignment(Proj)
    
    module_fine_alignment(Proj)