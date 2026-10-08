import streamlit as st

def module_parameter(Proj):
    st.markdown('<font color="blue">PARAMETERS</font>', unsafe_allow_html=True)
    
    with st.expander('Cylinder extraction'):
        num_params = len(Proj.parameters['cylinder_extraction'])
        columns = st.columns(3)
        for i, (k, v) in enumerate(Proj.parameters['cylinder_extraction'].items()):
            help_text = Proj.parameter_info.get(k, None)
            columns[i % 3].number_input(label=k, value=v, step=None, help=help_text, key=f'params_{k}')

    with st.expander('Fine alignment'):
        num_params = len(Proj.parameters['fine_alignment'])
        columns = st.columns(2)
        for i, (k, v) in enumerate(Proj.parameters['fine_alignment'].items()):
            help_text = Proj.parameter_info.get(k, None)
            columns[i % 3].number_input(label=k, value=v, step=None, help=help_text, key=f'params_{k}')
